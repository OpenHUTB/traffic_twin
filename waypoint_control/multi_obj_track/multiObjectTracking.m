% 数据路径
currentPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(currentPath, 'test_data');
% disp(dataPath)
% 获取所有 .mat 文件
matFiles = dir(fullfile(dataPath, "*.mat"));
[~, idx] = sort({matFiles.name});
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
ax = pcshow(ptCld);
showShape("Cuboid", lidarBboxes, Color="green", Parent=ax, Opacity=0.15, LineWidth=1);
zoom(ax, 8);

% 提取第一个相机数据
camIdx = 1;
[img, cameraBBoxes] = helperExtractCameraData(datalog, dataPath, camIdx);

 
% 在图像上插入边界框注释
img = insertObjectAnnotation(img, "Rectangle", cameraBBoxes, "Vehicle");
imshow(img);

% 设置跟踪器
tracker = trackerJPDA( ...
    TrackLogic="Integrated", ...
    FilterInitializationFcn=@helperInitLidarCameraFusionFilter, ...
    AssignmentThreshold=[20 200], ...
    MaxNumTracks=500, ...
    DetectionProbability=0.95, ...
    MaxNumEvents=50, ...
    ClutterDensity=1e-7, ...
    NewTargetDensity=1e-7, ...
    ConfirmationThreshold=0.99, ...
    DeletionThreshold=0.2, ...
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
% 路口1numFrames = 1076;
% 路口2
numFrames = 467;
% 设置固定的初始位置 (假设车辆静止于 ENU 坐标系的某点)
initialPosition = [0, 0, 0]; % 假设车辆在 ENU 原点

% 重复初始位置构成轨迹点
waypoints = repmat(initialPosition, numFrames, 1);

% 设置每个点的到达时间，模拟静止车辆的时间流逝
% 路口1 initialTime = 10.5462; % 初始时间 
% 路口2
initialTime = 22.4915;
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
        position = tracks(t).State([1, 3, 5]);  % 轨迹的位置 (x, y, z)
        velocity = tracks(t).State([2, 4, 6]);  % 轨迹的速度 (vx, vy, vz)
        
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

end

% 过滤掉轨迹数量少于20的车辆
allTracks = allTracks(cellfun(@(x) size(x, 1) >= 5, {allTracks.Positions}));
% 保存轨迹路径
savePath = fullfile(dataPath, 'trackedData.mat'); 

% 保存 allTracks 变量到 .mat 文件
save(savePath, 'allTracks');
disp(['轨迹数据已保存到 ', savePath]);
