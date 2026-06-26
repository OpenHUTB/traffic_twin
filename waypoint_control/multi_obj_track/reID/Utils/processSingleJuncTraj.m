% 链接单路口由于轨迹id变化导致的不连续轨迹
% 通过计算特征间的余弦相似度外还比较时间

function juncVehicleTraj = processSingleJuncTraj(trajStruct)  % 1x1 struct 
    threshold = 0.8;
    traj = trajStruct.traj;
    traj_f = trajStruct.traj_f;
    numTrajectories = length(traj);
    
    % 初始化一个 numTrajectories x numTrajectories 的矩阵来存储余弦相似度逻辑值
    cosineSimilarityMatrix = zeros(numTrajectories, numTrajectories);
    
    % 遍历所有轨迹对
    for i = 1:numTrajectories
        if isempty(traj{i})
            continue; 
        end
        cat_i = traj{i}.category;   % 获取轨迹 i 的类别
        for j = i+1:numTrajectories
            if isempty(traj{j})
               continue; 
            end
            cat_j = traj{j}.category;
            % 类别不同则不合并
            if ~isequal(cat_i, cat_j)
                continue;
            end
            % 获取轨迹 i 和 j 的位置数据（即特征向量 mean_hsv）
            pos_i = trajStruct.traj{i}.mean_hsv;
            pos_j = trajStruct.traj{j}.mean_hsv; 
            
            % 遇到空特征值，无法计算相似度，直接跳过本次匹配
            if isempty(pos_i) || isempty(pos_j)
                continue;
            end
            
            % 计算余弦相似度
            cosineSim = 1 - pdist2(pos_i, pos_j, 'cosine'); 
            if cosineSim > threshold
                % 存储余弦相似度
                cosineSimilarityMatrix(i, j) = 1;
                cosineSimilarityMatrix(j, i) = 1; % 因为余弦相似度是对称的
            end
        end
    end   
    % 将单个路口同一车辆的轨迹集合在一起
    groupedIndices = SingleJuncTraj(cosineSimilarityMatrix);
    % 删除空轨迹
    finalTraj = clearNoneTraj(groupedIndices, traj);
    % 同一车的多个轨迹保存航点多的
    %% 还需要处理同一车辆再次回到该路口
    finalTrajFiltered = filterFinalTraj(finalTraj, traj);
    % 保存最后包含时间的轨迹
    juncVehicleTraj = saveTraj(trajStruct, finalTrajFiltered);
end

function juncVehicleTraj = saveTraj(trajStruct, finalTrajMerged)
    % 保存最终数据，现在 finalTrajMerged 传进来的已经是拼接好的结构体了
    N = length(finalTrajMerged);
    
    % 预分配空间
    juncVehicleTraj.traj = cell(1, N);
    juncVehicleTraj.traj_f = zeros(N, 2); 
    
    for i = 1:N
        % 提取已经拼接好的数据
        mergedData = finalTrajMerged{i};
        
        % 直接赋值给最终的 cell
        juncVehicleTraj.traj{i} = mergedData;
        
        % 记录这条完整轨迹进入和离开路口的时间
        juncVehicleTraj.traj_f(i,:) = [mergedData.timestamp(1), mergedData.timestamp(end)];
    end 
end

function groupedIndices = SingleJuncTraj(qualifiedMatrix)
    % 将 qualifiedMatrix 转换为图的邻接矩阵
    adjMatrix = qualifiedMatrix;
    
    % 确保邻接矩阵是对称的
    adjMatrix = adjMatrix | adjMatrix';
    
    % 初始化并查集
    n = size(adjMatrix, 1); % 轨迹数量
    parent = 1:n; % 每个轨迹的父节点初始化为自己
    
    % 并查集的查找函数
    function root = find(u)
        while parent(u) ~= u
            parent(u) = parent(parent(u)); % 路径压缩
            u = parent(u);
        end
        root = u;
    end
    % 并查集的合并函数
    function union(u, v)
        rootU = find(u);
        rootV = find(v);
        if rootU ~= rootV
            parent(rootV) = rootU; % 将 rootV 的父节点设为 rootU
        end
    end
    % 遍历邻接矩阵，合并同一辆车的轨迹
    for i = 1:n
        for j = i+1:n
            if adjMatrix(i, j)
                union(i, j); % 如果 i 和 j 是同一辆车，合并它们
            end
        end
    end
    % 将同一辆车的轨迹索引分组
    groupedIndices = cell(n, 1); % 预分配空间
    for i = 1:n
        root = find(i);
        groupedIndices{root} = [groupedIndices{root}, i];
    end
    % 去除空单元格
    groupedIndices = groupedIndices(~cellfun(@isempty, groupedIndices));
end

function fianalTraj = clearNoneTraj(groupedIndices, traj)
    fianalTraj = groupedIndices;
    % 检查 traj 是否为空，并删除对应的索引
    validIndices = ~cellfun(@isempty, traj); % 找到 traj 中非空的索引
    for i = length(fianalTraj):-1:1
        % 保留 groupedIndices{i} 中在 validIndices 中为 true 的索引
        fianalTraj{i} = fianalTraj{i}(validIndices(fianalTraj{i}));
        
        % 如果 groupedIndices{i} 为空，则删除该组
        if isempty(fianalTraj{i})
            fianalTraj(i) = [];
        end
    end
end 

function finalTrajMerged = filterFinalTraj(finalTraj, traj)
    % 现在的逻辑：将同一组的碎片轨迹按时间顺序首尾拼接
    
    % 初始化结果，存放拼接后的新结构体
    finalTrajMerged = cell(1, length(finalTraj));
    
    % 遍历每一组需要合并的轨迹
    for i = 1:length(finalTraj)
        trajIndices = finalTraj{i};
        
        % 提取每段碎片的起始时间，用来排序（保证拼接顺序是从早到晚）
        startTimes = zeros(1, length(trajIndices));
        for j = 1:length(trajIndices)
            startTimes(j) = traj{trajIndices(j)}.timestamp(1);
        end
        [~, sortOrder] = sort(startTimes);
        sortedIndices = trajIndices(sortOrder); % 排序后的索引
        
        % 准备拼接容器
        merged_wrl_pos = [];
        merged_timestamp = [];
        
        % 为了保留最具代表性的特征，我们找到点数最多的一段，继承它的 trackID 和 mean_hsv
        maxPoints = -1;
        best_mean_hsv = [];
        best_trackID = [];
        
        for j = 1:length(sortedIndices)
            idx = sortedIndices(j);
            currentTraj = traj{idx};
            
            % 首尾纵向拼接位置和时间数据
            merged_wrl_pos = [merged_wrl_pos; currentTraj.wrl_pos];
            merged_timestamp = [merged_timestamp; currentTraj.timestamp];
            
            % 记录最长段的特征作为该车的代表特征
            if size(currentTraj.wrl_pos, 1) > maxPoints
                maxPoints = size(currentTraj.wrl_pos, 1);
                best_mean_hsv = currentTraj.mean_hsv;
                best_trackID = currentTraj.trackID;
            end
        end
        
        % 打包成新的完整结构体
        mergedStruct.trackID = best_trackID;
        mergedStruct.wrl_pos = merged_wrl_pos;
        mergedStruct.mean_hsv = best_mean_hsv;
        mergedStruct.timestamp = merged_timestamp;
        mergedStruct.category = traj{sortedIndices(1)}.category; % 类别取任意一个即可
        
        finalTrajMerged{i} = mergedStruct;
    end
end