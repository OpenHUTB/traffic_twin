function demoSingleJuncEvaluation(junc, juncNum)
    % 计算单个路口多个相机多目标跟踪精度（MOTA）
    currentPath = fileparts(mfilename('fullpath'));
    parentPath = fileparts(currentPath);
    truthsPath_vehicle = fullfile(parentPath, junc, 'vehicle_data/truths.mat');
    truthsData_vehicle = load(truthsPath_vehicle); 
    truthsPath_person = fullfile(parentPath, junc, 'pedestrian_data/truths.mat');
    truthsData_person = load(truthsPath_person); 
    % 路口真实车辆轨迹
    truthsStructured_data_vehicle = struct('Time', [], 'TruthID', [], 'Position', []);
    
    for i = 1:numel(truthsData_vehicle.truths)
        truthsStructured_data_vehicle(i).Time = double(truthsData_vehicle.truths{i}.Time);      
        truthsStructured_data_vehicle(i).TruthID = double(truthsData_vehicle.truths{i}.TruthID); 
        truthsStructured_data_vehicle(i).Position = double(truthsData_vehicle.truths{i}.Position); 
    end
    % 路口真实行人轨迹
    truthsStructured_data_person = struct('Time', [], 'TruthID', [], 'Position', []);
    
    for i = 1:numel(truthsData_person.truths)
        truthsStructured_data_person(i).Time = double(truthsData_person.truths{i}.Time);      
        truthsStructured_data_person(i).TruthID = double(truthsData_person.truths{i}.TruthID); 
        truthsStructured_data_person(i).Position = double(truthsData_person.truths{i}.Position); 
    end
    %%
    % 将两个结构体首尾相接拼成一个长条
    truthsStructured_data_all = [truthsStructured_data_vehicle(:)', truthsStructured_data_person(:)'];

    % 提取所有拼接后的时间戳
    all_times = [truthsStructured_data_all.Time];

    all_times_rounded = round(all_times, 3);

    % 对时间进行升序排序，拿到排序索引
    [~, sortIdx] = sort(all_times_rounded);

    % 用排序索引重新打乱排列整个结构体
    truthsStructured_data = truthsStructured_data_all(sortIdx);

    % 把结构体里面的时间也覆盖为消除误差后的干净时间
    for i = 1:numel(truthsStructured_data)
        truthsStructured_data(i).Time = all_times_rounded(sortIdx(i));
    end
    %%
    dirParts = strsplit(junc, filesep);
    fileName = [dirParts{2}, '_trackedTracks.mat'];
    tracksDataPath = fullfile(currentPath, dirParts{1}, fileName);
    % 跟踪到的路口车辆轨迹
    tracksData = load(tracksDataPath);
    tracksData = tracksData.evaluationTracks;
    
    for i = 1:numel(tracksData)
        tracksData(i).Position(3) = -0.7;
    end
    
    
    % 使用欧几里得距离评估 CLEAR MOT 指标
    metric = trackCLEARMetrics(SimilarityMethod="Euclidean",EuclideanScale=2);
    
    metricTable = evaluate(metric,tracksData,truthsStructured_data);
    avaluationName = [ dirParts{1}, '_Metric'];
    evaluationPath = fullfile(currentPath, avaluationName);
    if ~exist(evaluationPath, 'dir')
        mkdir(evaluationPath);
    end
    % 定义保存文件的路径
    metricTablePath = fullfile(evaluationPath, sprintf('metricTable_%d.mat', juncNum));
    
    % 保存表格到 MAT 文件
    save(metricTablePath, 'metricTable');
end