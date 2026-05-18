function detections = helperAssembleCameraDetections(cameraBox, cameraPose, time, sensorIdx, egoPose)
% helperAssembleCameraDetections transform camera bounding box detections into
% objectDetection format.

% This is a helper function and may be removed in a future release.

% Copyright 2022 The MathWorks, Inc.

if nargin<5
    egoPose = struct;
end

measurementParameters.SensorType = 2;
measurementParameters.SensorPose = cameraPose;
measurementParameters.EgoPose = egoPose;
measurementParameters.CameraIndex = sensorIdx - 1;

n = size(cameraBox,1);

sampleDetection = objectDetection(time, zeros(7,1),...
    'SensorIndex',sensorIdx,...
    'MeasurementNoise',1/(2*pi),...
    'MeasurementParameters',measurementParameters,...
    'ObjectAttributes',struct);

detections = repmat({sampleDetection},n,1);

for i = 1:size(cameraBox,1)
    detections{i}.Measurement(1:4) = cameraBox(i,:)/pixelScale();
    detections{i}.MeasurementNoise(1:4,1:4) = blkdiag(25/2,25/2,25,10)/(pixelScale());
    detections{i}.MeasurementNoise(5,5) = 4;
    if ~isempty(features) && i <= size(features, 1)
        % 取出第 i 个目标的特征向量
        detections{i}.ObjectAttributes = struct('Feature', features(i, :));
    else
        detections{i}.ObjectAttributes = struct('Feature', []);
    end
end

end