function demoSingleJuncEvalution(junc, juncNum)
    % 计算单个路口多个相机多目标跟踪精度（MOTA）
    currentPath = fileparts(mfilename('fullpath'));
    parentPath = fileparts(currentPath);
    truthsPath = fullfile(parentPath, junc, 'vehicle_data/truths.mat');
    truthsData = load(truthsPath); 
    % 路口真实车辆轨迹
    truthsStructured_data = struct('Time', [], 'TruthID', [], 'Position', []);
    
    for i = 1:numel(truthsData .truths)
        truthsStructured_data(i).Time = double(truthsData.truths{i}.Time);      
        truthsStructured_data(i).TruthID = double(truthsData.truths{i}.TruthID); 
        truthsStructured_data(i).Position = double(truthsData.truths{i}.Position); 
    end
    fileName = [junc, '_trackedTracks.mat'];
    tracksDataPath = fullfile(currentPath, 'demo_data', fileName);
    % 跟踪到的路口车辆轨迹
    tracksData = load(tracksDataPath);
    tracksData = tracksData.evaluationTracks;
    
    for i = 1:numel(tracksData)
        tracksData(i).Position(3) = -0.7;
    end
    
    
    % 使用欧几里得距离评估 CLEAR MOT 指标
    metric = trackCLEARMetrics(SimilarityMethod="Euclidean",EuclideanScale=2);
    
    metricTable = evaluate(metric,tracksData,truthsStructured_data);
    
    % 定义保存文件的路径
    metricTablePath = fullfile(currentPath, sprintf('metricTable_%d.mat', juncNum));
    
    % 保存表格到 MAT 文件
    save(metricTablePath, 'metricTable');
end