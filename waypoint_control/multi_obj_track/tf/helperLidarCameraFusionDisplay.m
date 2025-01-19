classdef helperLidarCameraFusionDisplay < matlab.System
    % helperLidarCameraFusionWithSmoothingDisplay A helper class to
    % visualize the detections and tracks.

    % This is a helper function and may be removed in a future release.

    % Copyright 2022 The MathWorks, Inc.


    properties
        ImagePlotters
        LidarDetectionPlotter
        LidarTrackPlotter
        CameraDetectionPlotter
        LidarPointCloudPlotter
        RecordGIF = true
    end

    properties(Access=protected)
        fig
        pFrames = {}
        DownsampleFactor = 5;
    end

    methods
        function obj = helperLidarCameraFusionDisplay(varargin)
            setProperties(obj, nargin, varargin{:});
        end
    end

    methods (Access = protected)
        function setupImpl(obj,dataFolder, datalog)
            obj.fig = figure('Units','normalized','Position',[0.01 0.03 0.98 0.9]);
            set(obj.fig,'visible','on');
            x = [0 0 1/3 1/3 2/3 2/3];
            dx = 1/3;
            % x = x/2;
            % dx = dx/2;
            y = [1/2 0 1/2 0 1/2 0];
            dy = 1/2;
            y = y/2;
            dy = dy/2;
            % idx = [3 5 1 2 4 6];
            idx = [2 1 3 4 5 6];
            imgPlotters = cell(6,1);
            % 显示相机图片
            camNames = {'front_camera','back_camera','front_left_camera','front_right_camera', 'left_camera','right_camera'};
            for i = 1:numel(idx)
                pi = uipanel('Parent',obj.fig);
                pi.Units = 'normalized';
                pi.Position = [x(i) y(i) dx dy];
                ax = axes('Parent',pi,'Units','normalized','Position',[0 0 1 1]);  
                % 原始路径
                originalPath = fullfile(dataFolder, datalog.CameraData(i).ImagePath);
            
                imgPlotters{i} = imshow(imread(originalPath),'Parent',ax);

                pi.Title = camNames{idx(i)};
            end


            obj.ImagePlotters = imgPlotters;
            p3d = uipanel('Parent',obj.fig,'Units','normalized','Position',[0 0.5 1 0.5]);
            p3d.Title = "lidar";
            ax = axes('Parent',p3d);
            points = datalog.LidarData.PointCloud.Location; 
            intensity = datalog.LidarData.PointCloud.Intensity; 
            
            % 创建 pointCloud 对象
            ptCloud = pointCloud(points, 'Intensity', intensity);
            pcshow(points,'Parent',ax);
            ax.XLim = [-60 100];
            ax.YLim = [-20 30];
            ax.ZLim = [-10 10];
 
            obj.LidarPointCloudPlotter = ax.Children(1);
            view(ax,0,90);
            tp = theaterPlot('Parent',ax);
            hold (ax,'on');
            obj.LidarTrackPlotter = trackPlotter(tp,'MarkerSize',4,'ColorizeHistory','off','ConnectHistory','off','MarkerFaceColor','green','MarkerEdgeColor','green');
            obj.LidarDetectionPlotter = trackPlotter(tp,'MarkerSize',2,'MarkerFaceColor','blue','MarkerEdgeColor','yellow');
            obj.CameraDetectionPlotter = trackPlotter(tp,'MarkerSize',2,'MarkerFaceColor','green','MarkerEdgeColor','blue');
            l = legend(ax);
            l.delete();

            %Add legends
            h2(1) = scatter(ax,NaN,NaN,20,'blue',"square","filled");
            h2(2) = scatter(ax,NaN,NaN,20,'yellow',"square","filled");
            h2(3) =  scatter(ax,NaN,NaN,20,'green',"square","filled");
            ax = obj.LidarPointCloudPlotter.Parent;
            legend(ax,h2,{'Camera Detections','Lidar Detections', 'Fused Tracks'}, 'location', 'eastoutside', 'FontSize',12,'TextColor','white','Color','black');
            obj.pFrames = {};

            %{
            figure
            ax = pcshow(ptCloud.Location);
            set(ax,'XLim',[-50 50],'YLim',[-40 40]);
            zoom(ax,2.5);
            axis off;
            %}
        end

        function releaseImpl(obj)
            % Release resources, such as file handles
        end

        function stepImpl(obj,dataFolder, dataLog, egoPose, lidarDetections, cameraDetections, tracks)
            idx = [1 2 3 4 5 6];
            if ~isempty(lidarDetections)
                d = [lidarDetections{:}];
                lidarMeas = horzcat(d.Measurement);
                pos = lidarMeas(1:3,:);
                vel = 0*pos;
                dim = lidarMeas(5:7,:);
                yaw = lidarMeas(4,:);
            else
                pos = zeros(3,0);
                vel = zeros(3,0);
                dim = zeros(3,0);
                yaw = zeros(1,0);
            end

            if ~isempty(cameraDetections)
                d = [cameraDetections{:}];
                cameraMeas = horzcat(d.Measurement);
                sensorIdx = horzcat(d.SensorIndex);
                cameraBox = cameraMeas(1:4,:)'*pixelScale();
            else
                cameraBox = zeros(4,0);
                sensorIdx =0;
            end

            if ~isempty(tracks)
                states = horzcat(tracks.State);
                trkPos = states([1 3 6],:);
                trkVel = states([2 4 7],:);
                trkDim = states([9 10 11],:);
                trkYaw = states(8,:);
                trkIds = horzcat(tracks.TrackID);
                labels = cellstr("T" + num2str(trkIds(:)));
                [trkPos, trkVel, trkDim, trkYaw] = transformForward(trkPos, trkVel, trkDim, trkYaw, egoPose);
            else
                trkPos = zeros(3,0);
                trkVel = zeros(3,0);
                trkDim = zeros(3,0);
                trkYaw = zeros(1,0);
                trkIds = zeros(1,0,'uint32');
                labels = cell(0,1);
            end

            for i = 1:numel(idx)
                % Plot camera image and detections
                 % 原始路径
                originalPath = fullfile(dataFolder,dataLog.CameraData(idx(i)).ImagePath);

                img = imread(originalPath);
                cameraBoxData = cameraBox(sensorIdx == idx(i) + 1, :);
                if isempty(cameraBoxData)
                    cameraBoxData = zeros(0, 4);  % 保证维度为 [0, 4]
                end
                img = insertObjectAnnotation(img,'rectangle',cameraBoxData,'C','Color','blue','LineWidth',2);

                % Plot lidar detections on this image
                cameraPose = dataLog.CameraData(idx(i)).Pose;
                camera = getMonoCamera(idx(i),cameraPose);

                % Project lidar data on camera
                [lidarBox, isValid] = cuboidProjection(camera, pos, dim, yaw);
                img = insertObjectAnnotation(img,'projected-cuboid',lidarBox(:,:,isValid),'L','Color','yellow','LineWidth',2);

                % Project track data on camera
                [trkBox, isValid] = cuboidProjection(camera, trkPos, trkDim, trkYaw);

                img = insertObjectAnnotation(img,'projected-cuboid',trkBox(:,:,isValid),labels(isValid),'Color','green','LineWidth',4,'FontSize',28);
                obj.ImagePlotters{i}.CData = img;
            end

            % Plot on 3-D plots
            set(obj.LidarPointCloudPlotter,...
                XData=dataLog.LidarData.PointCloud.Location(:,1),...
                YData=dataLog.LidarData.PointCloud.Location(:,2),...
                ZData=dataLog.LidarData.PointCloud.Location(:,3),...
                CData=dataLog.LidarData.PointCloud.Location(:,3));
            plotBox(obj, obj.LidarDetectionPlotter, pos, vel, dim, yaw);
            plotBox(obj, obj.LidarTrackPlotter, trkPos, trkVel, trkDim, trkYaw, labels);

            drawnow

            if obj.RecordGIF
                obj.pFrames{end+1} = getframe(obj.fig);
            end
        end
        function rotmat = rotz(obj,gamma)
            % rotate in the direction of x->y, counter-clockwise
            rotmat = [cosd(gamma) -sind(gamma) 0; sind(gamma) cosd(gamma) 0; 0 0 1];
        end
    end

    methods
        function plotBox(obj, plotter, pos, vel, dim, yaw, varargin)
            n = size(pos,2);
            dims = struct('Length',0,'Width',0,'Height',0,'OriginOffset',[0 0 0]);
            dims = repmat(dims,1,n);
            for i = 1:n
                dims(i).Length = dim(1,i);
                dims(i).Width = dim(2,i);
                dims(i).Height = dim(3,i);
                dims(i).OriginOffset = [0 0 0];
            end
            orient = repmat(eye(3),1,1,n);
            for i = 1:n
                orient(:,:,i) = rotz(obj,yaw(i))';
            end
            plotter.plotTrack(pos',dims,orient,varargin{:});
        end
    
        function writeAnimation(obj,fName)
            if obj.RecordGIF
                frames = obj.pFrames;
                imSize = size(frames{1}.cdata);
                im = zeros(imSize(1),imSize(2),1,floor(numel(frames)/obj.DownsampleFactor),'uint8');
                map = [];
                count = 1;
                for i = 1:obj.DownsampleFactor:numel(frames)
                    if isempty(map)
                        [im(:,:,1,count),map] = rgb2ind(frames{i}.cdata,256,'nodither');
                    else
                        im(:,:,1,count) = rgb2ind(frames{i}.cdata,map,'nodither');
                    end
                    count = count + 1;
                end
                imwrite(im,map,[fName,'.gif'],'DelayTime',0,'LoopCount',inf);
            end
        end
    end
end

function [projectedCuboids, isValid] = cuboidProjection(camera, pos, dim, yaw)
projectedCuboids = zeros(8,2,size(pos,2));
isValid = true(1,size(pos,2));
for i = 1:size(pos,2)
    projection = singleProjection(camera,pos(:,i),dim(:,i),yaw(i));
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
v = v([4 1 2 3 8 5 6 7],:);
v = v.*dim(:)';
orient = quaternion([yaw 0 0],'eulerd','ZYX','frame');
v = rotatepoint(orient, v);
v = v + pos(:)';
R = rotmat(quaternion([camera.Yaw camera.Pitch camera.Roll],'eulerd','ZYX','frame'),'frame');
p = [camera.SensorLocation camera.Height];
tform = rigid3d(R',p);
vCamera = transformPointsForward(tform,v);
[az,el] = cart2sph(vCamera(:,1),vCamera(:,2),vCamera(:,3));
[azFov, elFov] = computeFieldOfView(camera.Intrinsics.FocalLength,camera.Intrinsics.ImageSize);
inside = abs(az) < azFov/2 & abs(el) < elFov/2;
if sum(inside) > 4
    projectedCuboid = vehicleToImage(camera,v+[0 0 0.3158]);
else
    projectedCuboid = nan(8,2);
end

end

function [azFov, elFov] = computeFieldOfView(focalLength, imageSize)
azFov = 2*atan(imageSize(2)/(2*focalLength(1)));
elFov = 2*atan(imageSize(1)/(2*focalLength(2)));
end

function cameraBoxes = projectDataOnCamera(camera, pos, dim, yaw)
cameraBoxes = zeros(4,0);
for i = 1:size(pos,2)
    cameraBoxes = [cameraBoxes;projectOnCamera(camera,pos(:,i),dim(:,i),yaw(:,i))];
end
end

function cameraBox = projectOnCamera(camera, pos, dim, yaw)
v = [0.5000   -0.5000    0.5000
    0.5000    0.5000    0.5000
    -0.5000    0.5000    0.5000
    -0.5000   -0.5000    0.5000
    0.5000   -0.5000   -0.5000
    0.5000    0.5000   -0.5000
    -0.5000    0.5000   -0.5000
    -0.5000   -0.5000   -0.5000];
v = v.*dim(:)';

orient = quaternion([yaw 0 0],'eulerd','ZYX','frame');
v = rotatepoint(orient, v);
v = v + pos(:)';
R = rotmat(quaternion([camera.Yaw camera.Pitch camera.Roll],'eulerd','ZYX','frame'),'frame');
p = [camera.SensorLocation camera.Height];
tform = rigid3d(R',p);
vCamera = transformPointsForward(tform,v);
[az,el] = cart2sph(vCamera(:,1),vCamera(:,2),vCamera(:,3));
[azFov, elFov] = computeFieldOfView(camera.Intrinsics.FocalLength,camera.Intrinsics.ImageSize);
inside = abs(az) < azFov/2 & abs(el) < elFov/2;
if sum(inside) > 0
    cameraPoints = vehicleToImage(camera,v);
    u = min(cameraPoints(:,1));
    v = min(cameraPoints(:,2));
    w = max(cameraPoints(:,1)) - min(cameraPoints(:,1));
    h = max(cameraPoints(:,2)) - min(cameraPoints(:,2));
else
    u = 1e6;
    v = 1e6;
    w = 1e6;
    h = 1e6;
end
cameraBox = [u;v;w;h];
end

function colorOrder = darkColorOrder
colorOrder = [1.0000    1.0000    0.0667
    0.0745    0.6235    1.0000
    1.0000    0.4118    0.1608
    0.3922    0.8314    0.0745
    0.7176    0.2745    1.0000
    0.0588    1.0000    1.0000
    1.0000    0.0745    0.6510];

colorOrder(8,:) = [1 1 1];
colorOrder(9,:) = [0 0 0];
colorOrder(10,:) = 0.7*[1 1 1];
end