%% 车辆轨迹
function multiObjectTracking(junc, initTime, runFrameNum)

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
    % disp(class(datalog.LidarData.Pose.Orientation))
    % LidarData 字段，它包含了激光雷达数据。在该数据结构中，Timestamp 是记录该数据的时间戳。
    tOffset = datalog.LidarData.Timestamp;
    
    % 从 dataLog 中提取激光雷达点云数据和边界框。
    [ptCld, lidarBboxes] = helperExtractLidarData(datalog);
    
    % 显示点云数据
    disp(class(ptCld));
    pointCld = pointCloud(...
        ptCld.Location, ...          % 必需：N×3 坐标
        'Intensity', ptCld.Intensity ... % 可选：强度值
    );
    disp(class(pointCld));
    disp(pointCld);
    ax = pcshow(pointCld);
    showShape("Cuboid", lidarBboxes, Color="green", Parent=ax, Opacity=0.15, LineWidth=1);
    % zoom(ax, 'on'); % 启用缩放
    zoom(ax, 8);
    
    % 提取第一个相机数据
    camIdx = 1;
    [img, cameraBBoxes] = helperExtractCameraData(datalog, dataPath, camIdx);
    
     
    % 在图像上插入边界框注释
    img = insertObjectAnnotation(img, "Rectangle", cameraBBoxes, "Vehicle");
    imshow(img);
    
    tracker = trackerJPDA( ...
        TrackLogic="Integrated", ...
        FilterInitializationFcn=@helperInitLidarCameraFusionFilter, ...
        AssignmentThreshold=[5 30], ...   % 关联阈值
        MaxNumTracks=500, ...
        DetectionProbability=0.7, ...      % 检测到目标的概率
        MaxNumEvents=50, ...
        ClutterDensity=1e-7, ...
        NewTargetDensity=1e-6, ...
        ConfirmationThreshold=0.95, ...      % 确定为目标的概率
        DeletionThreshold=0.45, ...           % 表示一个跟踪目标被删除所需的最大置信度
        DeathRate=0.5);                     
                          
    % 初始化显示对象
    display = helperLidarCameraFusionDisplay;
    
    % 加载 GPS 数据
    % fileName = fullfile(dataPath, "gpsData.mat");
    % load(fileName, "gpsData");
    
    % 用于生成 ego 车辆轨迹
    % egoTrajectory = helperGenerateEgoTrajectory(gpsData);
    % disp(egoTrajectory)
    % 处理数据帧
    
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
        disp(['加载文件: ', fileName]);
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
            cameraBoxes{k} = camBBox; 
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
        disp(tracks)
    
        % 正常显示点云和3D框，将y坐标取反
        diversYDatalog = datalog;
        pointCloudLocation = diversYDatalog.LidarData.PointCloud.Location;
        pointCloudLocation(:, 2) = -pointCloudLocation(:, 2);  % 取反 Y 坐标
        diversYDatalog.LidarData.PointCloud.Location = pointCloudLocation;
    
         % 遍历 lidarDetections 中的每个元素
        for i = 1:numel(lidarDetections)
            % 取反 Measurement 中的第二个值
             lidarDetections{i}.Measurement(2) = -lidarDetections{i}.Measurement(2);
        end
        viewTracks = tracks;
        % 遍历 objectTrack 数组的每个元素
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
    
    %% 保存全部轨迹，用做计算指标
    % 轨迹文件夹
    dirParts = strsplit(junc, '/');
    allTracksFolderPath = fullfile(currentPath, 'Evaluation', dirParts{1});
    if ~exist(allTracksFolderPath, 'dir')
        mkdir(allTracksFolderPath);
    end
    fileName = [dirParts{2}, '_trackedTracks.mat'];
    allTracksPath = fullfile(allTracksFolderPath, fileName);
    save(allTracksPath, 'evaluationTracks');
    
    %% 保存部分较完整的轨迹，用作轨迹复现
    % 过滤掉轨迹数量少于5的车辆
    allTracks = allTracks(cellfun(@(x) size(x, 1) >= 5, {allTracks.Positions}));
    % 轨迹目录
    tracksDirectory = fullfile(dataPath, "tracks");
    if ~exist(tracksDirectory, 'dir')
        mkdir(tracksDirectory);
    end
    savePath = fullfile(tracksDirectory, 'trackedData.mat'); 
    % 保存 allTracks 变量到 .mat 文件
    save(savePath, 'allTracks');
    disp(['轨迹数据已保存到 ', savePath]);

end