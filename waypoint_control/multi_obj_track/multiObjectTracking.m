%% 车辆轨迹
function multiObjectTracking(junc, initTime, runFrameNum)

    % 数据路径
    currentPath = fileparts(mfilename('fullpath'));
    dataPath = fullfile(currentPath, junc);
    
    % 添加脚本路径
    folderPath = fullfile(currentPath, 'reID/Utils/misc');
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

    % 拆分字符串提取地图和路口名
    parts = strsplit(junc, '/');
    mapName = parts{1};   
    juncName = parts{2};  
    % 获取当前路口的配置参数
    params = getTrackerConfig(juncName);
    
    tracker = trackerJPDA( ...
        TrackLogic="Integrated", ...
        FilterInitializationFcn=@helperInitLidarCameraFusionFilter, ...
        AssignmentThreshold=params.AssignmentThreshold, ...   % 关联阈值
        MaxNumTracks=500, ...
        DetectionProbability=params.DetectionProbability, ...      % 检测到目标的概率
        MaxNumEvents=50, ...
        ClutterDensity=params.ClutterDensity, ...
        NewTargetDensity=params.NewTargetDensity, ...
        ConfirmationThreshold=params.ConfirmationThreshold, ...      % 确定为目标的概率
        DeletionThreshold=params.DeletionThreshold, ...           % 表示一个跟踪目标被删除所需的最大置信度
        DeathRate=params.DeathRate);                     
                          
    % 定义轨迹点的数量
    numFrames = runFrameNum;
    % 设置固定的初始位置 
    initialPosition = [0, 0, 0.98]; 
    
    % 重复初始位置构成轨迹点
    waypoints = repmat(initialPosition, numFrames, 1);
    
    % 设置每个点的到达时间，模拟静止车辆的时间流逝
    initialTime = initTime;
    timeInterval = 0.05;   % 每点的时间间隔
    timeOfArrival = initialTime + (0:numFrames-1)' * timeInterval;
    
    % 设置静止车辆的速度为零
    velocities = zeros(numFrames, 3);
    
    % 创建 `waypointTrajectory` 对象
    egoTrajectory = waypointTrajectory(waypoints, ...
        TimeOfArrival=timeOfArrival, ...
        Velocities=velocities, ...
        ReferenceFrame="ENU");
    
    % 初始化上一帧的时间变量
    prevTime = -inf;
    % 初始化一个结构体数组来保存每个目标的轨迹
    allTracks = struct('TrackID', {}, 'Positions', {}, 'Velocities', {}, 'Timestamps', {}, 'Features', {});
    evaluationTracks =  struct('Time', {}, 'TrackID', {}, 'Position', {});
    detectionsBool = false;
    
    % 开始逐帧处理
    for frame = 1:numFrames
        % 加载当前帧数据
        fileName = fullfile(dataPath, matFiles(frame).name);
        disp(['正在处理文件: ', fileName]);
        load(fileName, 'datalog');
        
        % 确保 LidarData.Pose.XXX 是 double 类型
        datalog.LidarData.Pose.Orientation = double(datalog.LidarData.Pose.Orientation);
        datalog.LidarData.Pose.Velocity = double(datalog.LidarData.Pose.Velocity);
        datalog.LidarData.Pose.Position = double(datalog.LidarData.Pose.Position);
        
        % 获取时间戳
        time = double(datalog.LidarData.Timestamp);
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
        [lidarBoxes, lidarPose] = helperExtractLidarData(datalog);
        % 转成 double
        lidarBoxes = double(lidarBoxes);
        if isempty(lidarBoxes) || size(lidarBoxes, 2) < 9
            lidarBoxes = zeros(0, 9);
        end
        lidarDetections = helperAssembleLidarDetections(lidarBoxes, lidarPose, time, 1, egoPose);
            
        % 提取相机检测数据
        cameraDetections = cell(0, 1);
        for k = 1:numel(datalog.CameraData)
            [camBBox, cameraPose] = helperExtractCameraData(datalog, dataPath, k);
            camBBox = double(camBBox);
            if isempty(camBBox) || size(camBBox, 2) < 4
                camBBox = zeros(0, 4);
            end
            if isfield(datalog.CameraData(k), 'Features')
                currentFeatures = double(datalog.CameraData(k).Features); 
            else
                currentFeatures = []; % 如果这帧没有特征，传空
            end
            thisCameraDetections = helperAssembleCameraDetections(camBBox, cameraPose, time, k + 1, egoPose, currentFeatures);
            cameraDetections = [cameraDetections; thisCameraDetections]; 
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

        % 送入跟踪器目标
        [tracks] = tracker(detections, time);

        camDetCell = cell(6, 1);  % 最多6个相机
        for d = 1:numel(cameraDetections)
            % 相机检测的 CameraIndex 是 (相机编号 + 1)
            camIdx = cameraDetections{d}.MeasurementParameters.CameraIndex - 1;
            if camIdx >= 1 && camIdx <= 6
                camDetCell{camIdx}{end+1} = cameraDetections{d};
            end
        end

        for t = 1:length(tracks)
            trackID = tracks(t).TrackID;
            position = tracks(t).State([1, 3, 6]);   % x,y,z
            velocity = tracks(t).State([2, 4, 7]);   % vx,vy,vz
            dim_track = tracks(t).State([9,10,11]);   % length,width,height
            yaw_track = tracks(t).State(8);           % yaw

            featureVec = [];  % 本帧该轨迹的特征

            % 尝试在每个相机中匹配投影框
            for camIdx = 1:numel(datalog.CameraData)
                % 获取相机姿态并构建相机对象
                cameraPose = datalog.CameraData(camIdx).Pose;
                camera = getMonoCamera(camIdx, cameraPose);

                % 将轨迹的3D框投影到该相机图像平面
                [projCuboid, isValid] = cuboidProjection(camera, position, dim_track, yaw_track);
                if ~isValid
                    continue;   % 目标不在这个相机的视野内
                end

                % 将该投影的8个顶点转为2D矩形框 [x, y, w, h]
                u = projCuboid(:,1);
                v = projCuboid(:,2);
                if any(isnan(u)) || any(isnan(v))
                    continue;
                end
                projBox = [min(u), min(v), max(u)-min(u), max(v)-min(v)];

                % 取该相机这一帧的所有2D检测框
                dets = camDetCell{camIdx};  % cell array
                if isempty(dets)
                    continue;
                end

                bestIoU = 0;
                bestDetIdx = 0;
                for d = 1:numel(dets)
                    meas = dets{d}.Measurement;
                    detBox = meas(1:4)';  % [u,v,w,h]
                    % 计算2D IoU
                    iou = bboxOverlapRatio(projBox, detBox, 'Min');  % 'Min' 对小目标更宽容
                    if iou > bestIoU
                        bestIoU = iou;
                        bestDetIdx = d;
                    end
                end

                % 若匹配程度足够高，则提取特征
                if bestIoU > 0.3  
                    matchedDet = dets{bestDetIdx};
                    if isfield(matchedDet.ObjectAttributes, 'Feature') && ...
                       ~isempty(matchedDet.ObjectAttributes.Feature)
                        featureVec = matchedDet.ObjectAttributes.Feature;
                        break;  % 找到特征，不再搜索其他相机
                    end
                end
            end

            % 将特征转为行向量
            if ~isempty(featureVec) && size(featureVec, 1) > 1
                featureVec = featureVec';
            end

            trackIdx = find([allTracks.TrackID] == trackID);
            if isempty(trackIdx)
                % 新建轨迹
                allTracks(end+1) = struct('TrackID', trackID, ...
                                          'Positions', position', ...
                                          'Velocities', velocity', ...
                                          'Timestamps', time, ...
                                          'Features', featureVec);
            else
                % 追加数据
                allTracks(trackIdx).Positions = [allTracks(trackIdx).Positions; position'];
                allTracks(trackIdx).Velocities = [allTracks(trackIdx).Velocities; velocity'];
                allTracks(trackIdx).Timestamps = [allTracks(trackIdx).Timestamps; time];

                % 若无特征则用 NaN 占位，保持行数一致
                if ~isempty(featureVec)
                    allTracks(trackIdx).Features = [allTracks(trackIdx).Features; featureVec];
                    % 动态记录特征维度
                    if ~exist('featureDim', 'var')
                        featureDim = length(featureVec);
                    end
                else
                    if exist('featureDim', 'var')
                        allTracks(trackIdx).Features = [allTracks(trackIdx).Features; nan(1, featureDim)];
                    end
                end
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
    if ~isempty(allTracks)
        allTracks = allTracks(cellfun(@(x) size(x, 1) >= 5, {allTracks.Positions}));
    end
    
    % 轨迹目录
    tracksDirectory = fullfile(dataPath, "tracks");
    if ~exist(tracksDirectory, 'dir')
        mkdir(tracksDirectory);
    end
    savePath = fullfile(tracksDirectory, 'trackedData.mat'); 
    % 保存 allTracks 变量到 .mat 文件
    save(savePath, 'allTracks');
    disp([' 轨迹提取完毕！数据已保存到 ', savePath]);

end


function [projectedCuboids, isValid] = cuboidProjection(camera, pos, dim, yaw)
    projectedCuboids = zeros(8,2,size(pos,2));
    isValid = true(1,size(pos,2));
    for i = 1:size(pos,2)
        projection = singleProjection(camera, pos(:,i), dim(:,i), yaw(i));
        projectedCuboids(:,:,i) = projection;
        if any(isnan(projection(:)))
            isValid(i) = false;
        end
    end
end

function projectedCuboid = singleProjection(camera, pos, dim, yaw)
    v = [0.5000   -0.5000    0.5000
         0.5000    0.5000    0.5000
        -0.5000    0.5000    0.5000
        -0.5000   -0.5000    0.5000
         0.5000   -0.5000   -0.5000
         0.5000    0.5000   -0.5000
        -0.5000    0.5000   -0.5000
        -0.5000   -0.5000   -0.5000];
    v = v([4 1 2 3 8 5 6 7], :);
    v = v .* dim(:)';
    orient = quaternion([yaw 0 0], 'eulerd', 'ZYX', 'frame');
    v = rotatepoint(orient, v);
    v = v + pos(:)';
    R = rotmat(quaternion([camera.Yaw camera.Pitch camera.Roll], 'eulerd', 'ZYX', 'frame'), 'frame');
    p = [camera.SensorLocation camera.Height];
    tform = rigid3d(R', p);
    vCamera = transformPointsForward(tform, v);
    [az, el] = cart2sph(vCamera(:,1), vCamera(:,2), vCamera(:,3));
    [azFov, elFov] = computeFieldOfView(camera.Intrinsics.FocalLength, camera.Intrinsics.ImageSize);
    inside = abs(az) < azFov/2 & abs(el) < elFov/2;
    if sum(inside) > 4
        projectedCuboid = vehicleToImage(camera, v + [0 0 0.3158]);
    else
        projectedCuboid = nan(8, 2);
    end
end

function [azFov, elFov] = computeFieldOfView(focalLength, imageSize)
    azFov = 2 * atan(imageSize(2) / (2 * focalLength(1)));
    elFov = 2 * atan(imageSize(1) / (2 * focalLength(2)));
end