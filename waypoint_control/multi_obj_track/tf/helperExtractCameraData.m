function [cameraBBoxes, pose, time] = helperExtractCameraData(dataLog, dataFolder, idx)
% This is a helper function and may be removed in a future release.

% Copyright 2022 The MathWorks, Inc.

pose = dataLog.CameraData(idx).Pose;
time = dataLog.CameraData(idx).Timestamp;
cameraBBoxes = dataLog.CameraData(idx).Detections;

end