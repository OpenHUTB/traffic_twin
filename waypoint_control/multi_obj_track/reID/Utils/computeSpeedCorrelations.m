function computeSpeedCorrelations(town)
    % 加载groundtruth 文件
    currentPath = fileparts(mfilename('fullpath'));   % 当前文件夹
    parentPath = fileparts(currentPath);              % 上级目录
    grandparentPath = fileparts(parentPath);          % 上上级目录
    
    targetDir = fullfile(grandparentPath, town);
    % 检查目标目录是否存在
    if ~isfolder(targetDir)
        error('目录不存在: %s', targetDir);
    end
    % 构建ground_truth.mat文件的完整路径
    groundTruthFile = fullfile(targetDir, 'ground_truth.mat');
    % 加载文件
    data = load(groundTruthFile);
    
    motdata = load('traj.mat');
    fieldNames = fieldnames(motdata);
    firstField = fieldNames{1}; % 获取第一个字段名
    cellArray = motdata.(firstField);
    correlations = [];
    % 遍历每个车辆的轨迹cell 
    for i = 1:length(cellArray)
        currentCell = cellArray{i}; % 获取第i个cell的内容
        % 获取当前cell中的第一个struct
        firstStruct = currentCell{1};  % 跟踪轨迹
        % 提取轨迹坐标数据
        detectedTraj = firstStruct.wrl_pos;
        detectedSpeed = firstStruct.speed;
        detectedTime = firstStruct.timestamp;
    
        trackPoints = detectedTraj;
    
        % 获取检测轨迹的时间范围
        startTime = min(detectedTime);
        endTime = max(detectedTime);
        lines = round((endTime - startTime)/0.05 + 1); % 截取行数
    
        % 遍历groudtruth找到最匹配的轨迹
        fieldNamesTruth = fieldnames(data);
        groundTruthField = fieldNamesTruth{1};
        groundTruthTrajectories = data.(groundTruthField);
    
        max_overlap = 0;
        best_truth_traj = [];
        best_start_index = 0;
    
        for j = 1:length(groundTruthTrajectories)
            currentTruth = groundTruthTrajectories{j};  % 真实轨迹
            Traj = currentTruth.wrl_pos;
            truth_points = Traj(:, 1:2);                % 只取x,y坐标
            n_truth_points = size(truth_points, 1);
            % 计算在真实轨迹中的起始行索引
            start_index = round((startTime - 0.05) / 0.05) + 1;
            % 检查起始索引是否有效
            if start_index < 1 || start_index > n_truth_points
                continue; % 起始索引超出范围
            end
    
            % 计算结束索引
            end_index = start_index + lines - 1;
    
             % 检查结束索引是否有效
            if end_index > n_truth_points
                continue; % 结束索引超出范围，这个真实轨迹不够长
            end
    
            % 截取对应时间段的真实轨迹
            matched_truth_points = truth_points(start_index:end_index, :);
            n_detected = size(trackPoints, 1);
            n_matched = size(matched_truth_points, 1);
    
            % 计算逐点误差
            tp = trackPoints(:,1:2);
            min_len = min(size(tp,1), size(matched_truth_points,1));
            aligned_track = tp(1:min_len, 1:2);
            aligned_truth = matched_truth_points(1:min_len, :);
            % 计算两个轨迹的初始点距离
            dist_start = norm(tp - matched_truth_points);
            if dist_start > 20
                continue;
            end
    
            [dtw_distance, ~] = dtw(aligned_track, aligned_truth);
            if ~isfinite(dtw_distance)
                continue;
            end
            point_errors = sqrt(sum((aligned_track - aligned_truth).^2, 2));
            % 计算最大可能距离
            max_single_error = max(point_errors);          % 最大单点误差
            max_distance = max_single_error * n_detected;  % 最大总距离
    
            % 计算重叠度
            if max_distance > 0
                overlap_ratio = 1 - (dtw_distance / max_distance);
                overlap_score = max(0, min(1, overlap_ratio));
            else
                overlap_score = 0;
            end
    
            % 更新最佳匹配
            if overlap_score > max_overlap
                max_overlap = overlap_score;
                best_truth_traj = Traj(start_index:end_index, :);
    
            end
      
        end
        if isempty(best_truth_traj)
           continue;
        end 
        detectedSpeed = sqrt(sum(detectedSpeed.^2, 2));
        truthSpeed = best_truth_traj(:,4);
        R = corrcoef(detectedSpeed, truthSpeed);
    
        % 提取相关系数值
        r_value = R(1, 2);
        correlations = [correlations; r_value];
    end
    
    %% 绘制箱型图
    first6 = town(1:6);
    % 计算均值和标准差
    mean_r = mean(correlations);
    std_r = std(correlations);
    
    figure('Color','w');
    boxplot(correlations, 'Notch', 'on', 'Colors', 'b', 'Whisker', 1.5);
    title(['Speed Curve Correlation (Pearson r) - ', first6]);  % Town10放标题
    ylabel('Correlation Coefficient (r)');
    xlabel('All Matched Trajectories');
    grid on;
    ylim([-1, 1]);
    % 坐标轴背景白色
    ax = gca;
    ax.Color = 'w';
    ax.GridColor = [0.8 0.8 0.8];
    ax.GridAlpha = 0.5;
    hold on;
    % 标注均值和标准差
    text(1.1, 0.2, sprintf('Mean = %.3f\nStd = %.3f', mean_r, std_r), ...
        'FontSize', 12, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
    hold off;

