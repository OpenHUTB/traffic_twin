%dataFolder = tempdir;
%dataFileName = "PandasetLidarCameraData.zip";
%url = "https://ssd.mathworks.com/supportfiles/driving/data/" + dataFileName;
%filePath = fullfile(dataFolder,dataFileName);
%if ~isfile(filePath)
%    websave(filePath,url);
%end
%unzip(filePath,dataFolder)
% 获取当前脚本所在的路径
currentFolder = fileparts(mfilename('fullpath'));
% 设置数据路径为当前脚本所在目录下的相对路径
dataPath = fullfile(currentFolder, 'PandasetLidarCameraData', 'PandasetLidarCameraData');
% 加载第一帧数据
fileName = fullfile(dataPath,strcat(num2str(1,"%03d"),".mat"));
% "dataLog": 这个参数指定加载文件中的变量 
load(fileName,"dataLog");

% LidarData 字段，它包含了激光雷达数据。在该数据结构中，Timestamp 是记录该数据的时间戳。
tOffset = dataLog.LidarData.Timestamp;
% 从 dataLog 中提取激光雷达点云数据和边界框。
[ptCld,lidarBboxes] = helperExtractLidarData(dataLog);
% 显示点云数据
ax = pcshow(ptCld);
% 显示 3D 边界框
showShape("Cuboid",lidarBboxes,Color="green",Parent=ax,Opacity=0.15,LineWidth=1);
zoom(ax,8)
% camIdx: 这是一个整数，表示你选择的相机的索引。由于代码中的 camIdx = 1，这表示选择第一个相机
camIdx = 1; 
% 提取相机数据
[img,cameraBBoxes] = helperExtractCameraData(dataLog,dataFolder,camIdx);
% 在图像上插入边界框注释
img = insertObjectAnnotation(img,"Rectangle",cameraBBoxes,"Vehicle");
% 显示图像
imshow(img)
% Setup  跟踪器 
tracker = trackerJPDA( ...
    TrackLogic="Integrated",...
    FilterInitializationFcn=@helperInitLidarCameraFusionFilter,...
    AssignmentThreshold=[20 200],...
    MaxNumTracks=500,...
    DetectionProbability=0.95,...
    MaxNumEvents=50,...
    ClutterDensity=1e-7,...
    NewTargetDensity=1e-7,...
    ConfirmationThreshold=0.99,...
    DeletionThreshold=0.2,...
    DeathRate=0.5);

% 初始化一个可视化对象，用于展示激光雷达（Lidar）和相机（Camera）数据融合的结果
display = helperLidarCameraFusionDisplay;

% 从 gpsData.mat 文件中加载名为 gpsData 的变量。
fileName = fullfile(dataPath,"gpsData.mat");
load(fileName,"gpsData");

% 用于根据 GPS 数据生成 ego 车辆的轨迹
egoTrajectory = helperGenerateEgoTrajectory(gpsData); 

% 处理多个数据帧，进行传感器数据的加载、融合、目标检测和跟踪。
% numFrames = 80 表示处理前 80 帧数据。
numFrames = 80;

for frame = 1:numFrames   
    % 加载当前帧的 .mat 文件
    fileName = fullfile(dataPath,strcat(num2str(frame,"%03d"),".mat"));
    load(fileName,'dataLog');

    % 获取当前帧的激光雷达时间戳，减去初始的时间偏移量 tOffset，得到当前的相对时间 time。
    time = dataLog.LidarData.Timestamp - tOffset;

    % 使用 GPS 数据生成的轨迹对象 egoTrajectory 查找当前时间点的 ego 车辆位置信息。
    % 返回位置 (pos)、姿态 (orient) 和速度 (vel)。
    [pos, orient, vel] = egoTrajectory.lookupPose(time);
    egoPose.Position = pos;
    egoPose.Orientation = eulerd(orient,"ZYX","frame");
    egoPose.Velocity = vel;

    % 从 dataLog 中提取激光雷达的检测数据，lidarBoxes 是雷达的检测边界框，lidarPose 是雷达的位置和姿态
    [~, lidarBoxes, lidarPose] = helperExtractLidarData(dataLog);
    % 将提取的雷达数据组装成目标检测格式
    lidarDetections = helperAssembleLidarDetections(lidarBoxes,lidarPose,time,1,egoPose);

    % 初始化一个空的单元格数组 cameraDetections，用于存储相机的检测数据。
    cameraDetections = cell(0,1);
    % 遍历所有的相机数据（dataLog.CameraData 是一个包含所有相机数据的结构）。
    for k = 1:1:numel(dataLog.CameraData)
        % 提取第 k 个相机的图像、边界框（camBBox）和相机姿态（cameraPose）。
        [img, camBBox,cameraPose] = helperExtractCameraData(dataLog, dataFolder,k);
        cameraBoxes{k} = camBBox; %#ok<SAGROW>
        % 将相机检测数据格式化为目标检测格式，并将其添加到 cameraDetections 中。
        thisCameraDetections = helperAssembleCameraDetections(cameraBoxes{k},cameraPose,time,k+1,egoPose);
        cameraDetections = [cameraDetections;thisCameraDetections]; %#ok<AGROW> 
    end

    % 如果是第一帧，仅使用激光雷达的检测结果 lidarDetections
    % 对于后续帧，将激光雷达和相机的检测结果拼接起来，得到完整的目标检测数据 detections。
    if frame == 1
        detections = lidarDetections;
    else
        detections = [lidarDetections;cameraDetections];
    end
    % 得到跟踪结果 tracks。
    tracks = tracker(detections, time);

    % Visualize the results
     display(dataFolder,dataLog, egoPose, lidarDetections, cameraDetections, tracks);
end