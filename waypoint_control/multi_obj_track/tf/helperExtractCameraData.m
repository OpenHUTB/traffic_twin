function [category, img, cameraBBoxes, pose, time] = helperExtractCameraData(dataLog, dataFolder, idx)
% This is a helper function and may be removed in a future release.

% Copyright 2022 The MathWorks, Inc.

imPath = fullfile(dataFolder,dataLog.CameraData(idx).ImagePath);
img = imread(imPath);
pose = dataLog.CameraData(idx).Pose;
time = dataLog.CameraData(idx).Timestamp;
cameraBBoxes = dataLog.CameraData(idx).Detections;
category = dataLog.CameraData(idx).Category;

end