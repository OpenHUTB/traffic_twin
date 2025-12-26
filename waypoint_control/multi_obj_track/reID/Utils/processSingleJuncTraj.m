% 链接单路口由于轨迹id变化导致的不连续轨迹
% 通过计算特征间的余弦相似度外还比较时间

function juncVehicleTraj = processSingleJuncTraj(trajStruct)  % 1x1 struct 
    threshold = 0.75;
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
        for j = i+1:numTrajectories
            if isempty(traj{j})
               continue; 
            end
            % 获取轨迹 i 和 j 的位置数据
            pos_i = trajStruct.traj{i}.mean_hsv;
            pos_j = trajStruct.traj{j}.mean_hsv; 
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

function juncVehicleTraj = saveTraj(trajStruct, finalTrajFiltered)
    tra = trajStruct.traj;
    N = length(finalTrajFiltered);
    traj_data = cell(1, N); 
    traj_f_data = zeros(N, 2); 
    % 将数据保存到结构体中
    juncVehicleTraj.traj = traj_data;
    juncVehicleTraj.traj_f = traj_f_data;
    
    for i = 1:N
    % 访问第 i 个元素
        cellElement = finalTrajFiltered{i};
        trackID = tra{cellElement}.trackID;
        positions = tra{cellElement}.wrl_pos;
        features = tra{cellElement}.mean_hsv;
        timeStamp = tra{cellElement}.timestamp;
        Category = tra{cellElement}.category;
        juncVehicleTraj.traj{i} = struct( ...
            'trackID', trackID, ...    % 轨迹 ID
            'wrl_pos', positions, ...  % 位置数据
            'mean_hsv', features, ...  % 特征数据
            'timestamp', timeStamp, ... % 轨迹时间
            'category', Category ... % 轨迹类别
        );
        juncVehicleTraj.traj_f(i,:) = [timeStamp(1), timeStamp(end)];
    end 
  
end

function groupedIndices = SingleJuncTraj(qualifiedMatrix)
    % 将 qualifiedMatrix 转换为图的邻接矩阵
    adjMatrix = qualifiedMatrix;
    
    % 确保邻接矩阵是对称的（因为 qualifiedMatrix 应该是无向图）
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


function finalTrajFiltered = filterFinalTraj(finalTraj, traj)
    % finalTraj: Nx1 cell，每个 cell 存放同一车辆的多个轨迹的索引
    % traj: 1xN cell，每个 cell 是一个 struct，struct 包含字段 wrl_pos（Mx3 double）
    
    % 初始化结果
    finalTrajFiltered = cell(size(finalTraj));
    
    % 遍历 finalTraj 中的每一组轨迹
    for i = 1:length(finalTraj)
        % 获取当前组的轨迹索引
        trajIndices = finalTraj{i};
        
        % 找到 wrl_pos 点数最多的轨迹
        maxPoints = -1; % 初始化最大点数
        bestIndex = -1; % 初始化最佳轨迹索引
        
        for j = 1:length(trajIndices)
            % 获取当前轨迹的 wrl_pos
            currentTraj = traj{trajIndices(j)};
            currentPoints = size(currentTraj.wrl_pos, 1); % wrl_pos 的行数
            
            % 如果当前轨迹的点数更多，则更新最佳轨迹
            if currentPoints > maxPoints
                maxPoints = currentPoints;
                bestIndex = trajIndices(j);
            end
        end
        
        % 将最佳轨迹索引存入结果
        finalTrajFiltered{i} = bestIndex;
    end
end