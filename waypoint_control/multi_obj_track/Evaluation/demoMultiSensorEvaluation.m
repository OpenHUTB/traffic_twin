function  MCTPResults = demoMultiSensorEvaluation(town)
    % 路口间多相机跟踪指标(MCTA)
    currentPath = fileparts(mfilename('fullpath'));
    parentPath = fileparts(currentPath);
    motaJunc1Path = fullfile(currentPath, town, 'metricTable_1.mat');
    numJunctions = 5;
    MCTPResults = zeros(1, numJunctions - 1); % 存储 MCTA 结果
    for i = 2:numJunctions 
        motaJunc2Path = fullfile(currentPath, town, sprintf('metricTable_%d.mat', i));
        
        motaJunc1 = load(motaJunc1Path);
        motaJunc2 = load(motaJunc2Path);
        
        motaJunc1Data = motaJunc1.metricTable;
        motaJunc2Data = motaJunc2.metricTable;
        GT1 = calculateGT(motaJunc1Data);
        GT2 = calculateGT(motaJunc2Data);
        
        idPath = fullfile(parentPath, 'reID/trkIDImg');
        junc1IDPath = fullfile(idPath,'test_data_junc1'); 
        junc2IDPath = fullfile(idPath,'test_data_junc2'); 
        datasetFolder = "trainedCustomReidNetwork.mat";
        netFolder = fullfile(parentPath, datasetFolder);
        data = load(netFolder);
        net = data.net;
        threshold = 0.65;
        IDSWinter = calculateIDSWinter(junc1IDPath, junc2IDPath, net, threshold);
        
        MCTP = calculateMCTP(GT1, GT2, motaJunc1Data, motaJunc2Data, IDSWinter);
        MCTPResults(i - 1) = MCTP; % 存储结果
    end 
end
function IDSWinter = calculateIDSWinter(IDPath1, IDPath2, net, threshold)
    features1 = extraFeatures(IDPath1, net);
    uniqueFeatures1 = filterUniqueImages(features1, threshold);

    features2 = extraFeatures(IDPath2, net);
    uniqueFeatures2 = filterUniqueImages(features2, threshold);
    numFeatures1 = size(uniqueFeatures1, 1);
    numFeatures2 = size(uniqueFeatures2, 1);
    % 初始化相似对计数器
    IDSWinter = 0;
    for i = 1:numFeatures1
        % 遍历第二个特征集中的每个特征向量
        for j = 1:numFeatures2
            % 计算余弦相似度
            cosSim = 1-pdist2(uniqueFeatures1(i, :), uniqueFeatures2(j, :),"cosine");
            if cosSim > threshold
                IDSWinter = IDSWinter + 1;
                break;
            end
        end
    end

end 

function features = extraFeatures(IDPath, net)
    imageFiles = dir(fullfile(IDPath, '*.jpeg')); % 或者 '*.jpeg', '*.png' 根据你的图片格式调整
    features = [];
    % 提取j文件夹中车辆的特征
    for i = 1:length(imageFiles)
        imgPath = fullfile(IDPath, imageFiles(i).name);
        img = imread(imgPath);
        feature = extractReidentificationFeatures(net,img);  
        features = [features; feature'];
    end

end

% 过滤函数
function uniqueFeatures = filterUniqueImages(features, threshold)
    numFeatures = size(features, 1); % 特征向量的数量
    retentionMask = true(numFeatures, 1); % 初始时假设所有行都要保留
    for i = 1:numFeatures
        for j = i+1:numFeatures
            % 计算余弦相似度
            cosSim = 1-pdist2(features(i, :), features(j, :),"cosine");
           
            % 如果相似度大于阈值，则标记其中一行为要删除（这里我们标记j行，可以随意选择）
            if cosSim > threshold
                retentionMask(j) = false; % 标记j行为不保留
            end
        end
    end
    % 根据标记保留特征向量
    uniqueFeatures = features(retentionMask, :);
end

function GT = calculateGT(motaJuncData)
    GT = (motaJuncData.("False Negative") + motaJuncData.("False Positive") + motaJuncData.("ID Switches"))/(1 - motaJuncData.("MOTA (%)")/100); 
end 


function MCTP = calculateMCTP(GT1, GT2, motaJunc1Data, motaJunc2Data, IDSWinter)
    FN1 = motaJunc1Data.("False Negative");
    FN2 = motaJunc2Data.("False Negative");
    
    FP1 = motaJunc1Data.("False Positive");
    FP2 = motaJunc2Data.("False Positive");
    
    IDSW1 = motaJunc1Data.("ID Switches");
    IDSW2 = motaJunc2Data.("ID Switches");

    MCTP = 1 - (FN1 + FN2 + FP1 + FP2 + IDSW1 + IDSW2 + IDSWinter)/(GT1 + GT2);
end