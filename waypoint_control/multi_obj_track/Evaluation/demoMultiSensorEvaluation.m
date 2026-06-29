function demoMultiSensorEvaluation(townPath)
    % 评估多路口下目标持续跟踪的整体精度与一致性（分人/车）
    % 输入 townPath: 例如 'Town01_Metric'
    % 要求存在 person_metricTable_i.mat / vehicle_metricTable_i.mat
    
    town = erase(townPath, "_Metric");
    currentPath = fileparts(mfilename('fullpath'));
    parentPath = fileparts(currentPath);
    
    categories = {'person', 'vehicle'};
    % 指定的路口对
    junctionPairs = [1,2; 1,3; 1,4; 1,5; 2,4; 2,5; 3,4; 3,5];
    
    for c = 1:length(categories)
        category = categories{c};
        fprintf('\n========== Evaluating Category: %s ==========\n', category);
        
        % 读取5个路口的单路口 MOTA ---
        MOTAList = zeros(1,5);
        for j = 1:5
            motaPath = fullfile(currentPath, townPath, ...
                                sprintf('%s_metricTable_%d.mat', category, j));
            data = load(motaPath);
            varName = sprintf('%smetricTable', category);
            MOTAList(j) = data.(varName).("MOTA (%)") / 100;   % 转为比例
        end
       
        % 遍历8个路口对，计算跨路口 MCTP 
        numPairs = size(junctionPairs, 1);
        MCTAResults = zeros(1, numPairs);
        
        for p = 1:numPairs
            j1 = junctionPairs(p, 1);
            j2 = junctionPairs(p, 2);
            
            % 读取两个路口的详细指标
            mota1Path = fullfile(currentPath, townPath, ...
                                 sprintf('%s_metricTable_%d.mat', category, j1));
            mota2Path = fullfile(currentPath, townPath, ...
                                 sprintf('%s_metricTable_%d.mat', category, j2));
            mota1 = load(mota1Path);
            mota2 = load(mota2Path);
            data1 = mota1.(varName);
            data2 = mota2.(varName);
            
            % 计算真实目标数 GT
            GT1 = calculateGT(data1);
            GT2 = calculateGT(data2);
            
            % 从轨迹中加载每个目标的特征
            features1 = loadTrackFeatures(j1, category, parentPath);
            features2 = loadTrackFeatures(j2, category, parentPath);
            
            % 跨路口 ID 匹配
            threshold = 0.5;
            IDSWinter = calculateIDSWinterFromFeatures(features1, features2, threshold);
            
            % 计算该路口对的 MCTP
            MCTAResults(p) = calculateMCTA(GT1, GT2, data1, data2, IDSWinter);
        end
        
        % 计算系统级 MCTA（用两个路口平均 MOTA 加权）
        MCTA = 0;
        for p = 1:numPairs
            j1 = junctionPairs(p,1);
            j2 = junctionPairs(p,2);
            meanMOTA = mean([MOTAList(j1), MOTAList(j2)]);
            MCTA = MCTA + meanMOTA * MCTAResults(p);
        end
        MCTA = MCTA / numPairs;
        
        % 输出结果
        fprintf('Per-junction MOTA: %s\n', num2str(round(MOTAList, 3)));
        fprintf('Pairwise MCTP (in order of pairs): %s\n', num2str(round(MCTAResults, 3)));
        fprintf('Overall MCTA: %.4f\n', MCTA);
    end
    fprintf('\n==============================================================\n');
end

function features = loadTrackFeatures(juncID, category, parentPath)
    % 加载指定路口、类别的轨迹特征，返回 N×24 矩阵
    featPath = fullfile(parentPath, 'reID','Utils','Town10HD_Opt', ...
                        sprintf('test_data_junc%d_%s_traj.mat', juncID, category));
    data = load(featPath);
    
    % 获取结构体的第一个字段名
    fieldName = fieldnames(data);
    firstField = fieldName{1};
    
    % cell 数组，每个元素是一个包含 'mean_hsv' 的结构体
    trackCells = data.(firstField).traj;
    
    numTracks = length(trackCells);
    features = zeros(numTracks, 24);   % 预分配
    
    for i = 1:numTracks
        feat = trackCells{i}.mean_hsv;
        if isequal(size(feat), [1, 24])
            features(i, :) = feat;
        else
            features(i, :) = zeros(1,24);
        end
    end
end

function IDSWinter = calculateIDSWinterFromFeatures(features1, features2, threshold)
    % 移除全零行（无效轨迹）
    valid1 = any(features1 ~= 0, 2);
    valid2 = any(features2 ~= 0, 2);
    f1 = features1(valid1, :);
    f2 = features2(valid2, :);
    
    % 归一化
    norm1 = sqrt(sum(f1.^2, 2));
    norm2 = sqrt(sum(f2.^2, 2));
    f1_norm = f1 ./ norm1;   % 全零行已在上面过滤，所以 norm1>0
    f2_norm = f2 ./ norm2;
    
    % 计算余弦相似度矩阵
    cosSimMatrix = f1_norm * f2_norm';
    
    % 统计匹配数：对每个 features1 中的有效轨迹，找在 features2 中第一个 > threshold 的匹配
    IDSWinter = 0;
    for i = 1:size(cosSimMatrix, 1)
        matches = find(cosSimMatrix(i, :) > threshold, 1);  % 找第一个匹配
        if ~isempty(matches)
            IDSWinter = IDSWinter + 1;
        end
    end
end

function GT = calculateGT(motaJuncData)
    % 从单路口指标反推真实目标数
    GT = (motaJuncData.("False Negative") + ...
          motaJuncData.("False Positive") + ...
          motaJuncData.("ID Switches")) / ...
         (1 - motaJuncData.("MOTA (%)") / 100);
end

function MCTP = calculateMCTA(GT1, GT2, motaJunc1Data, motaJunc2Data, IDSWinter)
    % 计算两路口间多摄像头跟踪精度
    FN1 = motaJunc1Data.("False Negative");
    FN2 = motaJunc2Data.("False Negative");
    FP1 = motaJunc1Data.("False Positive");
    FP2 = motaJunc2Data.("False Positive");
    IDSW1 = motaJunc1Data.("ID Switches");
    IDSW2 = motaJunc2Data.("ID Switches");
    
    MCTP = 1 - (FN1 + FN2 + FP1 + FP2 + IDSW1 + IDSW2 + IDSWinter) / (GT1 + GT2);
end