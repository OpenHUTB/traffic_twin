function traj = linkIdentities(juncTrajCell, matchThreshold)  % 1x5 cell
    % Town10场景中有5个路口，已经拿到每个路口的车辆轨迹，还包括其外观特征以及轨迹时间
    % 现需将5个路口的轨迹根据时间串联起来，形成完整的车辆轨迹
    threshold = matchThreshold;
    num_roads = length(juncTrajCell);
    % 用于存储匹配结果的容器
    matched_trajectories = containers.Map('KeyType', 'char', 'ValueType', 'any');

    persistent idCounter; % 定义一个持久的计数器，它会在脚本或函数的不同调用间保持其值
    if isempty(idCounter)
       idCounter = 0; % 首次使用时，将计数器初始化为0
    end
    % 遍历每个路口的数据
    for road_idx = 1:num_roads
        road_data = juncTrajCell{road_idx};
        traj_list = road_data.traj;
        
        % 遍历当前路口的每个车辆轨迹
        for traj_idx = 1:length(traj_list)
            current_traj = traj_list{traj_idx};
            current_features = current_traj.mean_hsv; % 提取外观特征
            
            % 尝试在当前结果集中找到匹配项
            is_matched = false;
            if road_idx > 1
                for key = keys(matched_trajectories)
                    matched_traj = matched_trajectories(key{1});
                    traj_struct = matched_traj{1};
                    % 这里需要一个函数来计算两个轨迹之间的相似度
                    % 基于外观特征的余弦相似度
                    similarity_score = 1 - pdist2(current_features,  traj_struct.mean_hsv, 'cosine'); 
                    
                    % 设定一个阈值来决定是否匹配
                    if similarity_score > threshold 
                        % 更新匹配轨迹
                        track = struct( ...
                            'roadID', road_idx, ...                 % 道路ID
                            'trackID', current_traj.trackID, ...    % 轨迹 ID
                            'wrl_pos', current_traj.wrl_pos, ...    % 位置数据
                            'mean_hsv', current_traj.mean_hsv, ...  % 特征数据
                            'timestamp', current_traj.timestamp ... % 轨迹时间
                        );
                        matched_traj{end+1} = track;
                        matched_trajectories(key{1}) = matched_traj;
                        is_matched = true;
                        break;
                    end
                end
            end
            % 如果没有找到匹配项，则作为新轨迹加入结果集
            if ~is_matched
                idCounter = idCounter + 1;
                currentTimeStamp = matlab_posixtime; 
                unique_id = sprintf('ID_%s_%06d', currentTimeStamp, idCounter);
                track = struct( ...
                    'roadID', road_idx, ...                 % 道路ID
                    'trackID', current_traj.trackID, ...    % 轨迹 ID
                    'wrl_pos', current_traj.wrl_pos, ...    % 位置数据
                    'mean_hsv', current_traj.mean_hsv, ...  % 特征数据
                    'timestamp', current_traj.timestamp ... % 轨迹时间
                );
                matched_trajectories(unique_id) = {track};
            end
        end
    end
    
    % 遍历集合
    keysList = keys(matched_trajectories);
    trajNum = length(keysList);
    trajCell = {};
    % 遍历所有的键
    for i = 1:trajNum
        key = keysList{i};           
        value = matched_trajectories(key); % 使用键来检索对应的值  
        trajCell{end+1} = value;
    end
    % 将匹配的轨迹按时间先后排序
    traj = sortByTimestamp(trajCell);

end

function posixTime = matlab_posixtime()
    % 获取当前日期和时间的 datenum 表示
    currentDateNum = datenum(now);
    
    % 将 datenum 转换为自 Unix 纪元以来的秒数
    % datenum 的基准是 0000-01-01，而 Unix 纪元是 1970-01-01
    % 因此，我们需要加上这两个日期之间的天数，并将其转换为秒
    daysToUnixEpoch = datenum(1970, 1, 1) - datenum(0, 0, 0);
    posixTime = (currentDateNum - daysToUnixEpoch) * 86400; % 一天有 86400 秒
end

function traj = sortByTimestamp(trajCell)
    traj = {};
    Ncell = length(trajCell);
    for i = 1:Ncell
        % 访问第 i 个单元数组元素
        cellElement = trajCell{i};
        N = length(cellElement);
        timestamps = arrayfun(@(x) x{1}.timestamp(1), cellElement);
        [~, sortIdx] = sort(timestamps);
        % 使用排序后的索引来重新排列 cell 数组中的结构体
        sortedCellElement = cell(1, N);
        if N > 1
           for j = 1:N
               sortedCellElement{j} = cellElement{sortIdx(j)};
           end
        end
        traj{end+1} = sortedCellElement;
    end 
    
end