
function computeDelay(town)
    % 加载groundtruth 文件
    currentPath = fileparts(mfilename('fullpath'));   % 当前文件夹
    parentPath = fileparts(currentPath);              % 上级目录
    grandparentPath = fileparts(parentPath);          % 上上级目录
    
    targetDir = fullfile(grandparentPath, 'Town10HD_Opt');
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
    all_detectedSpeeds = {};
    all_truthSpeeds = {};
    speed_real = [];
    speed_twin = [];
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

 
        % 计算最大行数
        maxLen = max([size(speed_twin,1), size(speed_real,1), length(detectedSpeed), length(truthSpeed)]);
        
        % 填充 speed_twin
        if size(speed_twin,1) < maxLen
            speed_twin(end+1:maxLen,:) = NaN;
        end
        
        % 填充 speed_real
        if size(speed_real,1) < maxLen
            speed_real(end+1:maxLen,:) = NaN;
        end
         
        % 填充新列 detectedSpeed
        if length(detectedSpeed) < maxLen
            detectedSpeed(end+1:maxLen,1) = NaN;
        end
        
        % 填充新列 truthSpeed
        if length(truthSpeed) < maxLen
            truthSpeed(end+1:maxLen,1) = NaN;
        end
        
        % 拼接
        speed_twin = [speed_twin, detectedSpeed];
        speed_real = [speed_real, truthSpeed];

    end 

    % ==========================
    % 输入
    % speed_real: N x M matrix (真实速度)，每列对应一辆车
    % speed_twin: N x M matrix (孪生速度)，对应关系一一匹配
    fs = 10; % 采样频率 (Hz) —— 如果你有时间向量 t, 可做 fs = 1/mean(diff(t));
    
    % =========================
    % 参数
    max_lag_seconds = 2;        % 最大允许搜索滞后（秒），防止估计过大噪声
    min_valid_samples = round(0.5 * fs);  % 序列至少要有多少有效样点才处理
    max_lag_samples = round(max_lag_seconds * fs);
    
    [N, M] = size(speed_real);
    lags_sec = nan(M,1);        % 存储每辆车估计的 lag（秒）
    r_before = nan(M,1);
    r_after  = nan(M,1);
    valid_mask = false(M,1);    % 标记哪些车有效
    
    for i = 1:M
        x = speed_twin(:,i);    % twin
        y = speed_real(:,i);    % real
    
        % 有效数据点足够否
        if sum(~isnan(x) & ~isnan(y)) < min_valid_samples
            continue;
        end
    
        % 1. 小段 NaN 插值（避免 xcorr 被 NaN 影响）
        if any(isnan(x))
            idx = ~isnan(x);
            x = interp1(find(idx), x(idx), (1:N)', 'linear', 'extrap');
        end
        if any(isnan(y))
            idx = ~isnan(y);
            y = interp1(find(idx), y(idx), (1:N)', 'linear', 'extrap');
        end
    
        % 2. 去均值（提高互相关稳定性）
        x0 = x - mean(x);
        y0 = y - mean(y);
    
        % 3. 互相关估计滞后（限制最大滞后）
        [c, lags] = xcorr(x0, y0, max_lag_samples, 'coeff'); % lags in samples
        [~, idxmax] = max(abs(c));
        lag_samples = lags(idxmax);
    
        % 注意符号约定：这里我们定义 positive lag_sec 表示 twin 落后于 real（twin 要向后移才对齐）
        lag_sec = -lag_samples / fs;  % 你可以通过单例验证符号是否符合你的定义
        lags_sec(i) = lag_sec;
    
        % 4. 用估计 lag 对齐 twin（向左/右移）
        shift = round(-lag_sec * fs); % shift >0 表示向左移动样本索引
        if shift > 0
            twin_aligned = [x(shift+1:end); nan(shift,1)];
        elseif shift < 0
            twin_aligned = [nan(-shift,1); x(1:end+shift)];
        else
            twin_aligned = x;
        end
    
        % 5. Pearson r before / after
        valid_before = ~isnan(x) & ~isnan(y);
        if sum(valid_before) > 3
            r_before(i) = corr(x(valid_before), y(valid_before));
        end
        valid_after = ~isnan(twin_aligned) & ~isnan(y);
        if sum(valid_after) > 3
            r_after(i) = corr(twin_aligned(valid_after), y(valid_after));
        end
    
        valid_mask(i) = true;
    end
    
    % ========== 统计与输出 ==========
    lags_used = lags_sec(valid_mask);
    r_before_used = r_before(valid_mask);
    r_after_used  = r_after(valid_mask);
    
    fprintf('Vehicles processed: %d / %d\n', sum(valid_mask), M);
    fprintf('Lag (s): mean=%.3f, std=%.3f, median=%.3f, P95=%.3f\n', ...
        nanmean(lags_used), nanstd(lags_used), nanmedian(lags_used), prctile(lags_used,95));
    fprintf('Pearson r before: mean=%.3f, std=%.3f\n', nanmean(r_before_used), nanstd(r_before_used));
    fprintf('Pearson r after align: mean=%.3f, std=%.3f\n', nanmean(r_after_used), nanstd(r_after_used));
    
    % ========== 可视化 ==========
    figure('Color','w','Position',[100 100 1000 600]);
    subplot(2,2,1);
    histogram(lags_used, 30);
    xlabel('Estimated lag (s)'); ylabel('Count'); title('Lag distribution'); grid on;
    
    subplot(2,2,2);
    boxplot([r_before_used; r_after_used], [zeros(size(r_before_used)); ones(size(r_after_used))], ...
        'Labels', {'Before','After'});
    title('Pearson r before vs after alignment');
    
    subplot(2,2,3);
    plot(sort(lags_used));
    xlabel('Vehicle index (sorted)'); ylabel('lag (s)'); title('Sorted lags');
    
    % 示例：画出第一个有效车辆的速度曲线（原始、延迟后的、对齐后）
    idx_example = find(valid_mask,1,'first');
    if ~isempty(idx_example)
        i = idx_example;
        x = speed_twin(:,i);
        y = speed_real(:,i);
        lag_sec = lags_sec(i);
        shift = round(-lag_sec * fs);
        if shift > 0
            twin_aligned = [x(shift+1:end); nan(shift,1)];
        elseif shift < 0
            twin_aligned = [nan(-shift,1); x(1:end+shift)];
        else
            twin_aligned = x;
        end
    
        subplot(2,2,4);
        t = (0:N-1)'/fs;
        plot(t, y, '-k', 'DisplayName','real'); hold on;
        plot(t, x, '--b', 'DisplayName','twin (orig)');
        plot(t, twin_aligned, ':r', 'DisplayName','twin (aligned)');
        legend('Location','best'); xlabel('Time (s)'); ylabel('Speed');
        title(sprintf('Example vehicle %d: lag=%.3fs, r_before=%.3f, r_after=%.3f', i, lag_sec, r_before(i), r_after(i)));
        grid on; hold off;
    end
end 