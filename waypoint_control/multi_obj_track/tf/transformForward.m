function [pos, vel, dim, yaw, posCov, velCov, yawCov, dimCov] = transformForward(pos, vel, dim, yaw, refPose, posCov, velCov, dimCov, yawCov)
% A function to move from parent frame to child frame. For example,
% scenario to ego, ego to sensor.

% refPose is the pose of the child with respect to its parent.

% This is a helper function and may be removed in a future release.

% Copyright 2022 The MathWorks, Inc.

if ~isempty(fieldnames(refPose))
    R = eul2rotm(deg2rad(refPose.Orientation(:)'))';
    pos = R'*(pos - refPose.Position(:));
    vel = R'*(vel - refPose.Velocity(:));
    egoYaw = refPose.Orientation(1);
    yaw = yaw - egoYaw;
    if nargin > 5
        posCov = R'*posCov*R;
        velCov = R'*velCov*R;
    end
end
end