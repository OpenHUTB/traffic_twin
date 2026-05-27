function detections = helperAssembleCameraDetections(cameraBox, cameraPose, time, sensorIdx, egoPose, features, currentCategory)
% helperAssembleCameraDetections transform camera bounding box detections into
% objectDetection format.

% This is a helper function and may be removed in a future release.

% Copyright 2022 The MathWorks, Inc.

if nargin<5
    egoPose = struct;
end

if nargin < 7
    currentCategory = [];
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
    
    attr = struct();
    
    % 填充Feature
    if ~isempty(features) && i <= size(features, 1)
        attr.Feature = features(i, :);
    else
        attr.Feature = [];
    end
    
    % 填充Category
    if ~isempty(currentCategory)
        % 若 currentCategory 的长度足够，取出对应元素，否则置空
        if i <= numel(currentCategory)
            attr.Category = currentCategory(i);
        else
            attr.Category = [];
        end
    else
        attr.Category = [];
    end
    
    detections{i}.ObjectAttributes = attr;

end

end