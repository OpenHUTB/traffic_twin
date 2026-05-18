function traj = linkIdentities(juncTrajCell, matchThreshold)  % 1x5 cell
    % Town10场景中有5个路口，已经拿到每个路口的车辆轨迹，还包括其外观特征以及轨迹时间
    % 现需将5个路口的轨迹根据时间串联起来，形成完整的车辆轨迹
    threshold = matchThreshold;
    num_roads = length(juncTrajCell);

    % 将所有路口的独立轨迹初始化为单独的“车辆链”容器
    allVehicles = {};
    for road_idx = 1:num_roads
        road_data = juncTrajCell{road_idx};
        % 兼容性检查：确保提取到轨迹列表
        if isstruct(road_data) && isfield(road_data, 'traj')
            traj_list = road_data.traj;
        else
            traj_list = road_data;
        end
        
        for traj_idx = 1:length(traj_list)
            current_traj = traj_list{traj_idx};
            
            % 统一规范化轨迹结构体
            track = struct( ...
                'roadID', road_idx, ...                 % 道路ID
                'trackID', current_traj.trackID, ...    % 轨迹 ID
                'wrl_pos', current_traj.wrl_pos, ...    % 位置数据
                'mean_hsv', current_traj.mean_hsv, ...  % 特征数据 (24维)
                'timestamp', current_traj.timestamp, ... % 轨迹时间
                'category', current_traj.category ...   % 轨迹类型
            );
            % 每个独立路口的轨迹，初始都视为一条独立的车辆轨迹链
            allVehicles{end+1} = {track};            
        end
    end

    % 关系定义：
    % 路口 1 可以去往：2, 3, 4, 5
    % 路口 2 可以去往：4, 5
    % 路口 3 可以去往：4, 5
    roadPairs = [1, 2;
                 1, 3;
                 1, 4;
                 1, 5;
                 2, 4;
                 2, 5;
                 3, 4;
                 3, 5];

    % 按照拓扑转移顺序，顺次使用 matchpairs 进行路口间的全局最优匹配
    for p = 1:size(roadPairs, 1)
        roadA = roadPairs(p, 1);
        roadB = roadPairs(p, 2);
        allVehicles = matchRoadPairs(allVehicles, roadA, roadB, threshold);
    end


    % isMultiJunction = cellfun(@(v) length(v) > 1, allVehicles);
    % allVehicles = allVehicles(isMultiJunction);

    % 转换回 trajCell 格式
    trajCell = allVehicles;

    % 将匹配的轨迹按时间先后排序
    trajByTime = sortByTimestamp(trajCell);
    % 删除时间重叠的轨迹
    traj = deleteOverlappingtimeTrajectories(trajByTime);
end

%%  基于匈牙利算法的路口对最优匹配 
function allVehicles = matchRoadPairs(allVehicles, roadA, roadB, threshold)
    % 寻找当前轨迹链末端在 roadA 的所有车辆
    idxA = [];
    for i = 1:length(allVehicles)
        if allVehicles{i}{end}.roadID == roadA
            idxA = [idxA, i];
        end
    end
    
    % 寻找当前轨迹链前端在 roadB 的所有车辆
    idxB = [];
    for j = 1:length(allVehicles)
        if allVehicles{j}{1}.roadID == roadB
            idxB = [idxB, j];
        end
    end
    
    % 如果任意一方没有车，则无需匹配
    if isempty(idxA) || isempty(idxB)
        return;
    end
    
    % 构建代价矩阵 (Cost Matrix)
    costMatrix = inf(length(idxA), length(idxB));
    
    for i = 1:length(idxA)
        vA = allVehicles{idxA(i)};
        trackA = vA{end}; % 取出 A 的最后一段路口轨迹
        
        % 计算车辆在路口 A 内部的行驶速度
        dx = abs(diff(trackA.wrl_pos(:, 1)));
        dy = abs(diff(trackA.wrl_pos(:, 2)));
        total_distance = sum(dx + dy);
        if length(trackA.timestamp) >= 2
            total_time = trackA.timestamp(end) - trackA.timestamp(1);
        else
            total_time = 0.05;
        end
        if total_time <= 0, total_time = 0.05; end
        speed = total_distance / total_time;
        
        last_position = trackA.wrl_pos(end, 1:2);
        last_time = trackA.timestamp(end);
        
        for j = 1:length(idxB)
            vB = allVehicles{idxB(j)};
            trackB = vB{1}; % 取出 B 的第一段路口轨迹
            
            f_position = trackB.wrl_pos(1, 1:2);
            f_time = trackB.timestamp(1);
            
            % 时间因果约束：路口 B 的出现时间必须在路口 A 结束之后
            d_time = f_time - last_time;
            if d_time <= 0
                continue;
            end
            
            % 计算路口间的转移速度
            dis = sum(abs(last_position - f_position));
            d_speed = dis / d_time;
            
            % 路口间的速度与时空物理约束判定
            if d_speed > speed / 2 && d_speed < speed * 2
                % 使用 24 维特征计算余弦距离
                dist_feat = pdist2(trackA.mean_hsv, trackB.mean_hsv, 'cosine');
                costMatrix(i, j) = dist_feat;
            end
        end
    end
    
    % 使用大局最优匹配函数 matchpairs (基于匈牙利算法)
    % 余弦距离 = 1 - 余弦相似度。 相似度 > threshold 意味着 距离 < 1 - threshold
    maxDistThresh = 1 - threshold;
    [matches, ~, ~] = matchpairs(costMatrix, maxDistThresh);
    
    % 融合匹配成功的车辆轨迹链
    if ~isempty(matches)
        vehiclesToRemove = false(1, length(allVehicles));
        
        for m = 1:size(matches, 1)
            globalIdxA = idxA(matches(m, 1));
            globalIdxB = idxB(matches(m, 2));
            
            % 将路口 B 的轨迹链顺次拼接到路口 A 后面，形成更长的跨路口链
            allVehicles{globalIdxA} = [allVehicles{globalIdxA}, allVehicles{globalIdxB}];
            % 标记原路口 B 的独立链为待删除
            vehiclesToRemove(globalIdxB) = true;
        end
        
        % 清理已被融合到长链中的重复项
        allVehicles = allVehicles(~vehiclesToRemove);
    end
end

function traj = deleteOverlappingtimeTrajectories(trajByTime)
    numSets = numel(trajByTime);
    newTrajByTime = cell(1,numSets);

    for i = 1:numSets
        trajectories = trajByTime{i};
        numTrajectories = numel(trajectories);
        if numTrajectories > 1
            valid = true(1, numTrajectories); % 初始化有效标记
            for j = 1:numTrajectories-1 % 注意这里是 numTrajectories-1
                if valid(j) % 如果当前轨迹有效
                    for k = j+1:numTrajectories % 从下一条轨迹开始检查重叠
                        if valid(k) % 如果下一条轨迹也有效
                            if trajectories{k}.timestamp(1) < trajectories{j}.timestamp(end)
                                valid(k) = false; % 标记为无效
                            end
                        end
                    end
                end
            end
            % 保留有效轨迹
            validTrajectories = trajectories(valid);
            newTrajByTime{i} = validTrajectories;
        else
            newTrajByTime{i} = trajectories;
        end
    end
    traj = newTrajByTime;
end

function posixTime = matlab_posixtime()
    currentDateNum = datenum(now);
    daysToUnixEpoch = datenum(1970, 1, 1) - datenum(0, 0, 0);
    posixTime = (currentDateNum - daysToUnixEpoch) * 86400; 
end

function traj = sortByTimestamp(trajCell)
    traj = {};
    Ncell = length(trajCell);
    for i = 1:Ncell
        cellElement = trajCell{i};
        N = length(cellElement);
        timestamps = arrayfun(@(x) x{1}.timestamp(1), cellElement);
        [~, sortIdx] = sort(timestamps);
        sortedCellElement = cell(1, N);
        for j = 1:N
            sortedCellElement{j} = cellElement{sortIdx(j)};
        end
        traj{end+1} = sortedCellElement;
    end 
end