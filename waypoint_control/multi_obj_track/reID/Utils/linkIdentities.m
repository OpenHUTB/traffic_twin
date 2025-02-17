function traj = linkIdentities(current_intersection_traj, next_intersection_traj)
    threshold = 0.5;
    % 路口的轨迹
    this_traj = current_intersection_traj.traj;
    next_traj = next_intersection_traj.traj;
    this_features = extractFeatures(this_traj);
    next_features = extractFeatures(next_traj);

    % 计算相似度矩阵，使用欧氏距离或余弦相似度
    similarity_matrix = computeSimilarityMatrix(this_features, next_features);
    % 使用相似度矩阵进行轨迹匹配
    matched_trajectories = matchTrajectories(similarity_matrix, threshold);
    

    ids1 = cell(1, numel(this_traj));  % 创建一个空 cell 数组来存储结果
    for i = 1:numel(this_traj)
         if isempty(this_traj{i})  % 如果该 cell 为空
             ids1{i} = -1;  % 返回 -1
         else
             ids1{i} = this_traj{i}.trackID;  % 否则提取 trackID
         end
    end
    ids1 = [ids1{:}]';  % 将 ids 转换为列向量

    ids2 = cell(1, numel(next_traj));  % 创建一个空 cell 数组来存储结果
    for i = 1:numel(next_traj)
         if isempty(next_traj{i})  % 如果该 cell 为空
             ids2{i} = -1;  % 返回 -1
         else
             ids2{i} = next_traj{i}.trackID;  % 否则提取 trackID
         end
    end
    ids2 = [ids2{:}]';  % 将 ids 转换为列向量

     % 合并匹配结果
    traj = mergeResults(this_traj, matched_trajectories, next_traj, ids1, ids2);

end

function features = extractFeatures(traj)
    % 假设每个轨迹的特征是一个 1x2048 的向量
    num_points = numel(traj);  % 轨迹点的数量
    features = zeros(num_points, 2048, 'single');  % 创建一个 1x2048 特征矩阵

    for i = 1:num_points
        if isempty(traj{i})  % 检查轨迹点是否为空
            features(i, :) = NaN(1, 2048);  % 如果空，返回 NaN 或者可以设置其他默认值
        else
            features(i, :) = traj{i}.mean_hsv;  % 提取轨迹点的特征，假设是 1x2048 的向量
        end
    end
end

function similarity_matrix = computeSimilarityMatrix(features1, features2)
    % 计算相似度矩阵，使用欧氏距离或余弦相似度
    num_traj_1 = size(features1, 1);
    num_traj_2 = size(features2, 1);
    similarity_matrix = zeros(num_traj_1, num_traj_2);

    for i = 1:num_traj_1
        for j = 1:num_traj_2
            % 计算余弦相似度为相似度度量
            similarity_matrix(i, j) = 1-pdist2(features1(i, :),features2(j, :),"cosine");
        end
    end
end

function matched_trajectories = matchTrajectories(similarity_matrix, threshold)
    % 使用相似度矩阵进行轨迹匹配
    % 如果相似度大于设定的阈值，则认为是匹配的
    matched_trajectories = similarity_matrix > threshold;
end

function traj = mergeResults(this_traj, matched_trajectories, next_traj, ids1, ids2)
    % 创建一个空的 cell 数组来存储合并后的轨迹
    merged_traj = {};  % 存储合并后的轨迹
    merged_ids = {};  % 存储合并后的轨迹 ID
    % 获取当前路口和下一个路口轨迹的数量
    num_current_trajectories = numel(this_traj);
    num_next_trajectories = numel(next_traj);
    % 遍历 matched_trajectories 矩阵，匹配为 1 的位置
    for i = 1:num_current_trajectories
        for j = 1:num_next_trajectories
            if matched_trajectories(i, j) == 1
                % 如果当前轨迹和下一个路口的轨迹匹配，合并这两个轨迹
                merged_traj{end+1} = mergeTwoTrajectories(this_traj{i}, next_traj{j});
                merged_ids{end+1} = {ids1(i), ids2(j)};  % 这将保存一个 cell，包含两个 ID
            end
        end
    end

     % 处理当前路口的轨迹没有匹配的情况
    for i = 1:num_current_trajectories
        % 如果当前轨迹没有与下一个路口的轨迹匹配，直接保留当前轨迹
        if all(matched_trajectories(i, :) == 0)  % 当前轨迹没有与任何下一个轨迹匹配
            merged_traj{end+1} = this_traj{i};
            merged_ids{end+1} = {ids1(i)};  % 只保存当前轨迹的 ID
        end
    end
    
    % 处理下一个路口的轨迹没有匹配的情况
    for j = 1:num_next_trajectories
        % 如果下一个路口的轨迹没有与任何当前轨迹匹配
        if all(matched_trajectories(:, j) == 0)  % 如果该轨迹没有与任何当前路口轨迹匹配
            merged_traj{end+1} = next_traj{j};  % 保留下一个路口的轨迹
            merged_ids{end+1} = {ids2(j) + num_current_trajectories};  % 更新下一个路口轨迹的 ID
        end
    end

     % 返回合并后的结果
    traj = struct('traj', merged_traj, 'ids', merged_ids);
end

function merged_trajectory = mergeTwoTrajectories(trajectory1, trajectory2)
    % 合并两个轨迹的功能函数
    merged_wrl_pos = [trajectory1.wrl_pos; trajectory2.wrl_pos];
    
    % 创建新的结构体来存储合并后的轨迹
    merged_trajectory = trajectory1;  % 复制第一个轨迹的信息
    
    % 更新合并后的 wrl_pos 字段
    merged_trajectory.wrl_pos = merged_wrl_pos;  % 合并后的位置信息
end