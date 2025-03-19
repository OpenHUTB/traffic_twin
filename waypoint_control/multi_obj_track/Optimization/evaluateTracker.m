function trackingError = evaluateTracker(params, junc, initTime, runFrameNum)
    % 数据路径
    currentPath = fileparts(mfilename('fullpath'));
    parentPath = fileparts(currentPath);
    dataPath = fullfile(parentPath, junc);

    % 获取所有 .mat 文件
    matFiles = dir(fullfile(dataPath, "*.mat"));
    % 提取文件名，并转换为数值数组用于排序
    fileNames = {matFiles.name};
    numericNames = cellfun(@(x) str2double(x), regexp(fileNames, '\d+', 'match'));
 
    % 对数字进行排序，并获取排序索引
    [~, idx] = sort(numericNames);
 
    % 根据排序索引重新排列 matFiles 结构体数组
    matFiles = matFiles(idx);
    % 创建 trackerJPDA 对象
    tracker = trackerJPDA(...
        TrackLogic="Integrated", ...
        FilterInitializationFcn=@helperInitLidarCameraFusionFilter, ...
        AssignmentThreshold=[5 30], ...
        MaxNumTracks=500, ...
        DetectionProbability=params.DetectionProbability, ...
        MaxNumEvents=50, ...
        ClutterDensity=params.ClutterDensity, ...
        NewTargetDensity=params.NewTargetDensity, ...
        ConfirmationThreshold=params.ConfirmationThreshold, ...
        DeletionThreshold=params.DeletionThreshold, ...
        DeathRate=params.DeathRate);
    % 定义轨迹点的数量
    numFrames = runFrameNum;
    % 设置固定的初始位置 (假设车辆静止于 ENU 坐标系的某点)
    initialPosition = [0, 0, 0.98]; % 假设车辆在 ENU 原点
    
    % 重复初始位置构成轨迹点
    waypoints = repmat(initialPosition, numFrames, 1);
    
    % 设置每个点的到达时间，模拟静止车辆的时间流逝
    initialTime = initTime;
    timeInterval = 0.05;   % 每点的时间间隔
    
    timeOfArrival = initialTime + (0:numFrames-1)' * timeInterval;
    
    % 设置静止车辆的速度为零
    velocities = zeros(numFrames, 3);
    % 设置方向为固定方向 (用四元数表示)
    orientation = quaternion(zeros(numFrames, 3), 'eulerd', 'ZYX', 'frame'); % 无旋转
    
    
    % 创建 `waypointTrajectory` 对象
    egoTrajectory = waypointTrajectory(waypoints, ...
        TimeOfArrival=timeOfArrival, ...
        Velocities=velocities, ...
        ReferenceFrame="ENU");
    
    % 手动设置其额外属性
    % egoTrajectory.Orientation = orientation;
    % 初始化上一帧的时间变量
    prevTime = -inf;
    % 初始化一个结构体数组来保存每个目标的轨迹
    allTracks = struct('TrackID', {}, 'Positions', {}, 'Velocities', {}, 'Timestamps', {});
    evaluationTracks =  struct('Time', {}, 'TrackID', {}, 'Position', {});
    detectionsBool = false;
    for frame = 1:numFrames
        % 加载当前帧数据
        fileName = fullfile(dataPath, matFiles(frame).name);
        load(fileName, 'datalog');
        % 确保 LidarData.Pose.Orientation 是 double 类型
    
        datalog.LidarData.Pose.Orientation = double(datalog.LidarData.Pose.Orientation);
        datalog.LidarData.Pose.Velocity = double(datalog.LidarData.Pose.Velocity);
        datalog.LidarData.Pose.Position = double(datalog.LidarData.Pose.Position);
        % 获取时间戳
        time = datalog.LidarData.Timestamp;
        if time <= prevTime
            warning('时间戳未递增，跳过帧 %d: 当前时间戳 %f', frame, time);
            continue;
        end
        % 更新上一帧时间
        prevTime = time;
        % 获取 ego 车辆的位置信息
        [pos, orient, vel] = egoTrajectory.lookupPose(egoTrajectory.TimeOfArrival(1));
        egoPose.Position = pos;
        egoPose.Orientation = eulerd(orient, "ZYX", "frame");
        egoPose.Velocity = vel;
    
        % 提取雷达检测数据
        [~, lidarBoxes, lidarPose] = helperExtractLidarData(datalog);
        lidarDetections = helperAssembleLidarDetections(lidarBoxes, lidarPose, time, 1, egoPose);
            
        % 提取相机检测数据
        cameraDetections = cell(0, 1);
        for k = 1:numel(datalog.CameraData)
            [img, camBBox, cameraPose] = helperExtractCameraData(datalog, dataPath, k);
            cameraBoxes{k} = camBBox; %#ok<SAGROW>
            thisCameraDetections = helperAssembleCameraDetections(cameraBoxes{k}, cameraPose, time, k + 1, egoPose);
            cameraDetections = [cameraDetections; thisCameraDetections]; %#ok<AGROW> 
        end
       
        % 合并检测结果
        if frame == 1
            detections = lidarDetections;
        else
            detections = [lidarDetections; cameraDetections];
        end

        if ~isempty(detections)
           detectionsBool = true;
        end 

        if ~detectionsBool
           continue;
        end

        % 跟踪目标
        tracks = tracker(detections, time);
     
        % 更新所有目标的轨迹
        for t = 1:length(tracks)
            trackID = tracks(t).TrackID;
            position = tracks(t).State([1, 3, 6]);  % 轨迹的位置 (x, y, z)
            velocity = tracks(t).State([2, 4, 7]);  % 轨迹的速度 (vx, vy, vz)
            
            % 检查该 TrackID 是否已存在于 allTracks 中
            trackIdx = find([allTracks.TrackID] == trackID);
            
            if isempty(trackIdx)
                % 如果该 TrackID 尚未记录，则创建一个新的轨迹
                allTracks(end + 1) = struct('TrackID', trackID, ...
                                             'Positions', position', ...  % 转置为行向量，确保每列分别为 x, y, z
                                             'Velocities', velocity', ...  % 转置为行向量，确保每列分别为 vx, vy, vz
                                             'Timestamps', time);
            else
                % 如果该 TrackID 已存在，则更新该轨迹
                allTracks(trackIdx).Positions = [allTracks(trackIdx).Positions; position'];
                allTracks(trackIdx).Velocities = [allTracks(trackIdx).Velocities; velocity'];
                allTracks(trackIdx).Timestamps = [allTracks(trackIdx).Timestamps; time];
            end
        end
    
        for t = 1:length(tracks)
            trackID = tracks(t).TrackID;
            position = tracks(t).State([1, 3, 6]);  
            evaluationTracks(end+1).Time = time;
            evaluationTracks(end).TrackID = trackID;
            evaluationTracks(end).Position = position;
        end
    
    end
    %% 计算指标
    % 检查 tracks 是否为空
    if isempty(evaluationTracks)
        warning('tracks 为空，跳过当前一轮评估。');
        trackingError = NaN; % 返回 NaN 表示无效评估
        return;
    end

    truthsPath = fullfile(parentPath, junc, 'vehicle_data/truths.mat');
    truthsData = load(truthsPath); 
    % 路口真实车辆轨迹
    truthsStructured_data = struct('Time', [], 'TruthID', [], 'Position', []);  
    for i = 1:numel(truthsData .truths)
        truthsStructured_data(i).Time = double(truthsData.truths{i}.Time);      
        truthsStructured_data(i).TruthID = double(truthsData.truths{i}.TruthID); 
        truthsStructured_data(i).Position = double(truthsData.truths{i}.Position); 
    end
    % 跟踪到的路口车辆轨迹
    for i = 1:numel(evaluationTracks)
        evaluationTracks(i).Position(3) = -0.7;
    end   
    % 使用欧几里得距离评估 CLEAR MOT 指标
    metric = trackCLEARMetrics(SimilarityMethod="Euclidean", EuclideanScale=2);
    metricTable = evaluate(metric, evaluationTracks, truthsStructured_data);
    trackingError = 1 - metricTable{:, 1}; % 最小化 MOTA 的补数
end