%% 行人轨迹
function multiobjecttracking(junc, initTime, runFrameNum)

    % 数据路径
    currentPath = fileparts(mfilename('fullpath'));
    dataPath = fullfile(currentPath, junc);
    
    % 添加脚本路径
    folderPath = fullfile(currentPath, 'reID/Utils/misc');
    % 将文件夹添加到路径
    addpath(folderPath);
    % 获取所有 .mat 文件
    matFiles = dir(fullfile(dataPath, "*.mat"));
    % 提取文件名，并转换为数值数组用于排序
    fileNames = {matFiles.name};
    numericNames = cellfun(@(x) str2double(x), regexp(fileNames, '\d+', 'match'));
 
    % 对数字进行排序，并获取排序索引
    [~, idx] = sort(numericNames);
 
    % 根据排序索引重新排列 matFiles 结构体数组
    matFiles = matFiles(idx);
    
    % 加载第一帧数据
    fileName = fullfile(dataPath, matFiles(1).name);
    load(fileName, "datalog");
    tOffset = datalog.LidarData.Timestamp;
    
    % 从 dataLog 中提取激光雷达点云数据和行人边界框
    ptCld = pointCloud(datalog.LidarData.PointCloud.Location);
    
    % 提取行人边界框（基于尺寸过滤）
    lidarBboxes = [];
    if isfield(datalog.LidarData, 'Objects')
        allBboxes = datalog.LidarData.Objects.BoundingBox;
        for i = 1:size(allBboxes, 1)
            bbox = allBboxes(i, :);
            dims = bbox(4:6); % 长宽高
            % 行人尺寸判断（宽<1m，高>1.5m）
            if dims(2) < 1 && dims(3) > 1.5
                lidarBboxes = [lidarBboxes; bbox];
            end
        end
    end
    
    % 显示点云数据
    ax = pcshow(ptCld);
    if ~isempty(lidarBboxes)
        showShape("Cuboid", lidarBboxes, Color="green", Parent=ax, Opacity=0.15, LineWidth=1);
    end
    zoom(ax, 8);
    
    % 提取第一个相机数据
    camIdx = 1;
    if isfield(datalog, 'CameraData') && numel(datalog.CameraData) >= camIdx
        camData = datalog.CameraData(camIdx);
        imgPath = fullfile(dataPath, camData.FileName);
        img = imread(imgPath);
        
        % 提取行人检测框
        cameraBBoxes = [];
        if isfield(camData, 'Detections')
            for i = 1:length(camData.Detections)
                if strcmpi(camData.Detections(i).Class, 'pedestrian')
                    cameraBBoxes = [cameraBBoxes; camData.Detections(i).BoundingBox];
                end
            end
        end
        
        % 在图像上插入边界框注释
        if ~isempty(cameraBBoxes)
            img = insertObjectAnnotation(img, "Rectangle", cameraBBoxes, "Pedestrian");
        end
        imshow(img);
    end
    
    % 行人专用跟踪器配置
    tracker = trackerJPDA( ...
        TrackLogic="Integrated", ...
        FilterInitializationFcn=@initPedestrianFilter, ...
        AssignmentThreshold=[3 15], ...   % 降低关联阈值适应行人运动
        MaxNumTracks=200, ...             % 减少最大跟踪数
        DetectionProbability=0.6, ...     % 行人检测概率
        MaxNumEvents=30, ...
        ClutterDensity=1e-6, ...
        NewTargetDensity=1e-5, ...
        ConfirmationThreshold=0.90, ...   % 降低确认阈值
        DeletionThreshold=0.40, ...       % 提高删除灵敏度
        DeathRate=0.7);                   % 提高消失率
    
    % 初始化显示对象
    display = helperPedestrianTrackingDisplay;
    
    % 定义轨迹点的数量
    numFrames = runFrameNum;
    % 设置固定的初始位置 (假设车辆静止)
    initialPosition = [0, 0, 0.98]; % 假设车辆在 ENU 原点
    
    % 重复初始位置构成轨迹点
    waypoints = repmat(initialPosition, numFrames, 1);
    
    % 设置每个点的到达时间
    initialTime = initTime;
    timeInterval = 0.05;   % 每点的时间间隔
    
    timeOfArrival = initialTime + (0:numFrames-1)' * timeInterval;
    
    % 设置静止车辆的速度为零
    velocities = zeros(numFrames, 3);
    % 设置方向为固定方向
    orientation = quaternion(zeros(numFrames, 3), 'eulerd', 'ZYX', 'frame');
    
    % 创建 `waypointTrajectory` 对象
    egoTrajectory = waypointTrajectory(waypoints, ...
        TimeOfArrival=timeOfArrival, ...
        Velocities=velocities, ...
        ReferenceFrame="ENU");
    
    % 初始化上一帧的时间变量
    prevTime = -inf;
    % 初始化轨迹结构
    allTracks = struct('TrackID', {}, 'Positions', {}, 'Velocities', {}, 'Timestamps', {});
    evaluationTracks = struct('Time', {}, 'TrackID', {}, 'Position', {});
    detectionsBool = false;
    
    for frame = 1:numFrames
        % 加载当前帧数据
        fileName = fullfile(dataPath, matFiles(frame).name);
        disp(['加载文件: ', fileName]);
        load(fileName, 'datalog');
        
        % 确保数据类型正确
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
    
        % 提取雷达检测数据（行人专用）
        lidarDetections = {};
        if isfield(datalog.LidarData, 'Objects')
            for i = 1:length(datalog.LidarData.Objects)
                obj = datalog.LidarData.Objects(i);
                % 行人判断（通过尺寸或类别）
                if (obj.BoundingBox(4) < 1 && obj.BoundingBox(6) > 1.5) || ...
                   isfield(obj, 'Class') && strcmpi(obj.Class, 'pedestrian')
                    detection = objectDetection(time, ...
                        [obj.BoundingBox(1:3)'; zeros(3,1)], ... % 位置和速度
                        'SensorIndex', 1, ...
                        'ObjectClassID', 2); % 2表示行人
                    lidarDetections{end+1} = detection;
                end
            end
        end
            
        % 提取相机检测数据（行人专用）
        cameraDetections = cell(0, 1);
        if isfield(datalog, 'CameraData')
            for k = 1:numel(datalog.CameraData)
                camData = datalog.CameraData(k);
                if isfield(camData, 'Detections')
                    for i = 1:length(camData.Detections)
                        det = camData.Detections(i);
                        if strcmpi(det.Class, 'pedestrian')
                            % 简化的2D到3D转换（假设地面高度为0）
                            pos3D = [det.BoundingBox(1:2), 0]';
                            detection = objectDetection(time, ...
                                pos3D, ...
                                'SensorIndex', k+1, ...
                                'ObjectClassID', 2);
                            cameraDetections{end+1} = detection;
                        end
                    end
                end
            end
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

        % 跟踪行人目标
        tracks = tracker(detections, time);
        
        % 调整坐标系显示
        diversYDatalog = datalog;
        pointCloudLocation = diversYDatalog.LidarData.PointCloud.Location;
        pointCloudLocation(:, 2) = -pointCloudLocation(:, 2);
        diversYDatalog.LidarData.PointCloud.Location = pointCloudLocation;
    
        for i = 1:numel(lidarDetections)
            lidarDetections{i}.Measurement(2) = -lidarDetections{i}.Measurement(2);
        end
        
        viewTracks = tracks;
        for i = 1:numel(viewTracks)
            viewTracks(i).State(3, :) = -viewTracks(i).State(3, :);     
        end
        
        % 可视化结果
        display(dataPath, diversYDatalog, egoPose, lidarDetections, cameraDetections, viewTracks);
        
        % 更新所有目标的轨迹
        for t = 1:length(tracks)
            trackID = tracks(t).TrackID;
            position = tracks(t).State([1, 3, 6]);  % 轨迹的位置 (x, y, z)
            velocity = tracks(t).State([2, 4, 7]);  % 轨迹的速度 (vx, vy, vz)
            
            % 检查该 TrackID 是否已存在于 allTracks 中
            trackIdx = find([allTracks.TrackID] == trackID);
            
            if isempty(trackIdx)
                % 创建新的行人轨迹
                allTracks(end + 1) = struct('TrackID', trackID, ...
                                           'Positions', position', ...
                                           'Velocities', velocity', ...
                                           'Timestamps', time);
            else
                % 更新现有行人轨迹
                allTracks(trackIdx).Positions = [allTracks(trackIdx).Positions; position'];
                allTracks(trackIdx).Velocities = [allTracks(trackIdx).Velocities; velocity'];
                allTracks(trackIdx).Timestamps = [allTracks(trackIdx).Timestamps; time];
            end
        end
    
        % 保存评估数据
        for t = 1:length(tracks)
            evaluationTracks(end+1).Time = time;
            evaluationTracks(end).TrackID = tracks(t).TrackID;
            evaluationTracks(end).Position = tracks(t).State([1, 3, 6]);
        end
    end
    
    %% 保存全部轨迹，用做计算指标
    dirParts = strsplit(junc, '/');
    allTracksFolderPath = fullfile(currentPath, 'Evaluation', dirParts{1});
    if ~exist(allTracksFolderPath, 'dir')
        mkdir(allTracksFolderPath);
    end
    fileName = [dirParts{2}, '_pedestrianTracks.mat'];
    allTracksPath = fullfile(allTracksFolderPath, fileName);
    save(allTracksPath, 'evaluationTracks');
    
    %% 保存部分较完整的轨迹，用作轨迹复现
    % 过滤掉轨迹数量少于10的行人（行人需要更长的轨迹）
    allTracks = allTracks(cellfun(@(x) size(x, 1) >= 10, {allTracks.Positions}));
    % 轨迹目录
    tracksDirectory = fullfile(dataPath, "pedestrian_tracks");
    if ~exist(tracksDirectory, 'dir')
        mkdir(tracksDirectory);
    end
    savePath = fullfile(tracksDirectory, 'trackedPedestrians.mat'); 
    % 保存 allTracks 变量到 .mat 文件
    save(savePath, 'allTracks');
    disp(['行人轨迹数据已保存到 ', savePath]);

end

%% 行人滤波器初始化函数
function filter = initPedestrianFilter(detection)
    % 使用恒定速度模型
    filter = initcvekf(detection);
    
    % 调整过程噪声适应行人运动
    filter.ProcessNoise = diag([1, 1, 1, 0.5, 0.5, 0.5]); % 行人运动更随机
    
    % 调整状态转换模型
    filter.StateTransitionModel = [...
        1 0 0 1 0 0;  % x = x + vx
        0 1 0 0 1 0;  % y = y + vy
        0 0 1 0 0 1;  % z = z + vz
        0 0 0 1 0 0;  % vx = vx
        0 0 0 0 1 0;  % vy = vy
        0 0 0 0 0 1]; % vz = vz
    
    % 调整测量噪声
    filter.MeasurementNoise = eye(3)*0.2; % 行人检测噪声可能更大
end