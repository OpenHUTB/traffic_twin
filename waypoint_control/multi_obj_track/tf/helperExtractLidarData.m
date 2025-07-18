function [ptCloud, lidarBoxes, pose, time] = helperExtractLidarData(dataLog)
% This is a helper function and may be removed in a future release.

% Copyright 2022 The MathWorks, Inc.

% ptCloud = dataLog.LidarData.PointCloud;
   % 提取点的位置和强度信息
points = dataLog.LidarData.PointCloud.Location; 
intensity = dataLog.LidarData.PointCloud.Intensity; 
% 创建 pointCloud 对象
ptCloud = pointCloud(points, 'Intensity', intensity);

pose = dataLog.LidarData.Pose;
time = dataLog.LidarData.Timestamp;
lidarBoxes = dataLog.LidarData.Detections;

end