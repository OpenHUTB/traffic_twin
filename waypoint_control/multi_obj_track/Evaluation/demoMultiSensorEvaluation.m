function demoMultiSensorEvaluation(townPath)
    % 评估多路口（multi-intersection）下目标持续跟踪的整体精度与一致性。
    % 我们计算的是路口1 与其余4个路口之间的值
  
    town = erase(townPath, "_Metric");
    % town = 'Town01';
    % new_str = [town, '_Metric'];
    currentPath = fileparts(mfilename('fullpath'));
    parentPath = fileparts(currentPath);
    numJunctions = 5;

    % --- 1. 初始化结果 ---
    MCTAResults = zeros(1, numJunctions - 1);
    MOTAList = zeros(1, numJunctions);

    % --- 2. 读取各路口的单路口指标 ---
    for i = 1:numJunctions
        motaPath = fullfile(currentPath, townPath, sprintf('metricTable_%d.mat', i));
        motaData = load(motaPath);
        MOTAList(i) = motaData.metricTable.("MOTA (%)") / 100;  % 转成比例
    end

    % --- 3. 遍历相邻路口对，计算跨路口 MCTP ---
    motaJunc1Path = fullfile(currentPath, townPath, 'metricTable_1.mat');
    for i = 2:numJunctions
        motaJunc2Path = fullfile(currentPath, townPath, sprintf('metricTable_%d.mat', i));

        motaJunc1 = load(motaJunc1Path);
        motaJunc2 = load(motaJunc2Path);
        motaJunc1Data = motaJunc1.metricTable;
        motaJunc2Data = motaJunc2.metricTable;

        GT1 = calculateGT(motaJunc1Data);
        GT2 = calculateGT(motaJunc2Data);

        % --- ReID路径和模型 ---
        idPath = fullfile(parentPath, 'reID/trkIDImg');
        junc1IDPath = fullfile(idPath, sprintf('%s/test_data_junc%d', town, i-1)); 
        junc2IDPath = fullfile(idPath, sprintf('%s/test_data_junc%d', town, i));
        datasetFolder = "trainedCustomReidNetwork.mat";
        netFolder = fullfile(parentPath, datasetFolder);
        data = load(netFolder);
        net = data.net;

        % --- 跨路口ID匹配 ---
        threshold = 0.5;
        IDSWinter = calculateIDSWinter(junc1IDPath, junc2IDPath, net, threshold);

        % --- 计算MCTP ---
        MCTA = calculateMCTA(GT1, GT2, motaJunc1Data, motaJunc2Data, IDSWinter);
        MCTAResults(i - 1) = MCTA;

        % 更新下一个基准路口
        motaJunc1Path = motaJunc2Path;
    end

    % --- 4. 计算系统级 MCTA ---
    MCTA = 0;
    for i = 1:length(MCTAResults)
        meanMOTA = mean([MOTAList(i), MOTAList(i+1)]);
        MCTA = MCTA + meanMOTA * MCTAResults(i);
    end
    MCTA = MCTA / length(MCTAResults);

    % --- 5. 输出结果 ---
    fprintf('================ Multi-Intersection Evaluation ================\n');
    fprintf('Town: %s\n', town);
    fprintf('Per-junction MOTA: %s\n', num2str(round(MOTAList, 3)));
    fprintf('Pairwise MCTP: %s\n', num2str(round(MCTAResults, 3)));
    fprintf('Overall MCTA: %.4f\n', MCTA);
    fprintf('==============================================================\n');
end


% ======================= 子函数部分 =======================

function IDSWinter = calculateIDSWinter(IDPath1, IDPath2, net, threshold)
    features1 = extraFeatures(IDPath1, net);
    uniqueFeatures1 = filterUniqueImages(features1, threshold);

    features2 = extraFeatures(IDPath2, net);
    uniqueFeatures2 = filterUniqueImages(features2, threshold);

    numFeatures1 = size(uniqueFeatures1, 1);
    numFeatures2 = size(uniqueFeatures2, 1);

    IDSWinter = 0;
    for i = 1:numFeatures1
        for j = 1:numFeatures2
            cosSim = 1 - pdist2(uniqueFeatures1(i, :), uniqueFeatures2(j, :), "cosine");
            if cosSim > threshold
                IDSWinter = IDSWinter + 1;
                break;
            end
        end
    end
end


function features = extraFeatures(IDPath, net)
    imageFiles = dir(fullfile(IDPath, '*.jpeg'));
    features = [];
    for i = 1:length(imageFiles)
        imgPath = fullfile(IDPath, imageFiles(i).name);
        img = imread(imgPath);
        feature = extractReidentificationFeatures(net, img);
        features = [features; feature'];
    end
end


function uniqueFeatures = filterUniqueImages(features, threshold)
    numFeatures = size(features, 1);
    retentionMask = true(numFeatures, 1);
    for i = 1:numFeatures
        for j = i+1:numFeatures
            cosSim = 1 - pdist2(features(i, :), features(j, :), "cosine");
            if cosSim > threshold
                retentionMask(j) = false;
            end
        end
    end
    uniqueFeatures = features(retentionMask, :);
end


function GT = calculateGT(motaJuncData)
    GT = (motaJuncData.("False Negative") + motaJuncData.("False Positive") + motaJuncData.("ID Switches")) / ...
        (1 - motaJuncData.("MOTA (%)") / 100);
end


function MCTP = calculateMCTA(GT1, GT2, motaJunc1Data, motaJunc2Data, IDSWinter)
    FN1 = motaJunc1Data.("False Negative");
    FN2 = motaJunc2Data.("False Negative");

    FP1 = motaJunc1Data.("False Positive");
    FP2 = motaJunc2Data.("False Positive");

    IDSW1 = motaJunc1Data.("ID Switches");
    IDSW2 = motaJunc2Data.("ID Switches");

    MCTP = 1 - (FN1 + FN2 + FP1 + FP2 + IDSW1 + IDSW2 + IDSWinter) / (GT1 + GT2);
end
