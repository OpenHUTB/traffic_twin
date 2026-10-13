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
    % 过滤掉 Position 为空的无效真值条目（无 TruthID 且无位置信息）
    truthsStructured_data_vehicle = truthsStructured_data_vehicle(~cellfun(@isempty, {truthsStructured_data_vehicle.Position}));
    % 路口真实行人轨迹
    truthsStructured_data_person = struct('Time', [], 'TruthID', [], 'Position', []);
    
    for i = 1:numel(truthsData_person.truths)
        truthsStructured_data_person(i).Time = double(truthsData_person.truths{i}.Time);
        truthsStructured_data_person(i).TruthID = double(truthsData_person.truths{i}.TruthID);
        truthsStructured_data_person(i).Position = double(truthsData_person.truths{i}.Position);
    end
    % 过滤掉 Position 为空的无效真值条目（无 TruthID 且无位置信息）
    truthsStructured_data_person = truthsStructured_data_person(~cellfun(@isempty, {truthsStructured_data_person.Position}));
    % %%
    % % 将两个结构体首尾相接拼成一个长条
    % truthsStructured_data_all = [truthsStructured_data_vehicle(:)', truthsStructured_data_person(:)'];
    % 
    % % 提取所有拼接后的时间戳
    % all_times = [truthsStructured_data_all.Time];
    % 
    % all_times_rounded = round(all_times, 3);
    % 
    % % 对时间进行升序排序，拿到排序索引
    % [~, sortIdx] = sort(all_times_rounded);
    % 
    % % 用排序索引重新打乱排列整个结构体
    % truthsStructured_data = truthsStructured_data_all(sortIdx);
    % 
    % % 把结构体里面的时间也覆盖为消除误差后的干净时间
    % for i = 1:numel(truthsStructured_data)
    %     truthsStructured_data(i).Time = all_times_rounded(sortIdx(i));
    % end
    %% 获取人车跟踪轨迹地址
    % dirParts = strsplit(junc, filesep);
    % fileName = [dirParts{2}, '_trackedTracks.mat'];
    % tracksDataPath = fullfile(currentPath, dirParts{1}, fileName);
    % % 跟踪到的路口车辆轨迹
    % tracksData = load(tracksDataPath);
    % tracksData = tracksData.evaluationTracks;
    % 
    % for i = 1:numel(tracksData)
    %     tracksData(i).Position(3) = -0.7;
    % end
    dirParts = strsplit(junc, filesep);
    vehiclefile = [dirParts{2}, '_vehicle_trackedTracks.mat'];
    vehicletracksDataPath = fullfile(currentPath, dirParts{1}, vehiclefile);
    personfile = [dirParts{2}, '_person_trackedTracks.mat'];
    persontracksDataPath = fullfile(currentPath, dirParts{1}, personfile);

    vehicletracksData = load(vehicletracksDataPath);
    vehicletracksData = vehicletracksData.vehicleevaluationTracks;
    persontracksData = load(persontracksDataPath);
    persontracksData = persontracksData.personevaluationTracks;

    % 仅保留落在有效真值时间范围内的轨迹（真值记录结束后仍产生的轨迹视为超出评估范围）
    validVehicleTimes = unique([truthsStructured_data_vehicle.Time]);
    vehicletracksData = vehicletracksData(ismembertol([vehicletracksData.Time], validVehicleTimes));
    validPersonTimes = unique([truthsStructured_data_person.Time]);
    persontracksData = persontracksData(ismembertol([persontracksData.Time], validPersonTimes));

    for i = 1:numel(vehicletracksData)
        vehicletracksData(i).Position(3) = -0.8;
    end
    for i = 1:numel(persontracksData)
        persontracksData(i).Position(3) = -0.8;
    end
    
    
    % 使用欧几里得距离评估 CLEAR MOT 指标
    metric = trackCLEARMetrics(SimilarityMethod="Euclidean",EuclideanScale=2);
    vehiclemetricTable = evaluate(metric,vehicletracksData,truthsStructured_data_vehicle);
    personmetricTable = evaluate(metric,persontracksData,truthsStructured_data_person);

    % 计算 HOTA 指标（检测精度 + 关联精度 + 定位精度的综合）
    vehiclehotaTable = evaluateHOTA(vehicletracksData, truthsStructured_data_vehicle);
    personhotaTable = evaluateHOTA(persontracksData, truthsStructured_data_person);

    avaluationName = [ dirParts{1}, '_Metric'];
    evaluationPath = fullfile(currentPath, avaluationName);
    if ~exist(evaluationPath, 'dir')
        mkdir(evaluationPath);
    end
    % 定义保存文件的路径
    vehiclemetricTablePath = fullfile(evaluationPath, sprintf('vehicle_metricTable_%d.mat', juncNum));
    personmetricTablePath = fullfile(evaluationPath, sprintf('person_metricTable_%d.mat', juncNum));
    vehiclehotaTablePath = fullfile(evaluationPath, sprintf('vehicle_hotaTable_%d.mat', juncNum));
    personhotaTablePath = fullfile(evaluationPath, sprintf('person_hotaTable_%d.mat', juncNum));

    % 保存表格到 MAT 文件
    save(vehiclemetricTablePath, 'vehiclemetricTable');
    save(personmetricTablePath, 'personmetricTable');
    save(vehiclehotaTablePath, 'vehiclehotaTable');
    save(personhotaTablePath, 'personhotaTable');

    % 打印结果
    fprintf('===== 路口 %d 车辆 CLEAR 指标 =====\n', juncNum);
    disp(vehiclemetricTable);
    fprintf('===== 路口 %d 车辆 HOTA 指标 =====\n', juncNum);
    disp(vehiclehotaTable);
    fprintf('===== 路口 %d 行人 CLEAR 指标 =====\n', juncNum);
    disp(personmetricTable);
    fprintf('===== 路口 %d 行人 HOTA 指标 =====\n', juncNum);
    disp(personhotaTable);
end