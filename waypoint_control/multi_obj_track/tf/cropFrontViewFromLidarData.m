function [croppedPointCloudObj, processedLabels] = cropFrontViewFromLidarData(lidarData, boxLabels, pcRange)
% This function crops the front view from the input full-view point cloud
% and also processes the corresponding box labels according to the 
% specified grid parameters.

    xmin = pcRange(1,1);
    xmax = pcRange(1,2);
    ymin = pcRange(1,3);
    ymax = pcRange(1,4);
    zmin = pcRange(1,5);
    zmax = pcRange(1,6);
    
    tmpStr = '';
    % 获取 boxLabels 的行数
    numFiles = size(boxLabels,1);
    
    processedLabels = cell(size(boxLabels));
    croppedPointCloudObj = cell(size(numFiles));

    % Get the classes from the ground truth labels.
    classNames = boxLabels.Properties.VariableNames;
    
    for i = 1:numFiles

        ptCloud = read(lidarData);  
        % 获取当前文件的标签数据。
        groundTruth = boxLabels(i,:);
        
        % Set the limits for the point cloud.
        if(numel(size(ptCloud.Location)) == 3)
            % Organized point cloud
            [x,y] = find( ptCloud.Location(:,:,1) < xmax ...
                                & ptCloud.Location(:,:,1) > xmin ...
                                & ptCloud.Location(:,:,2) < ymax ...
                                & ptCloud.Location(:,:,2) > ymin ...
                                & ptCloud.Location(:,:,3) < zmax ...
                                & ptCloud.Location(:,:,3) > zmin);    
            ptCloud = select(ptCloud, x, y, 'OutputSize', 'full'); 
        else
            % Unorganized point cloud
            pos = find( ptCloud.Location(:,1) < xmax ...
                & ptCloud.Location(:,1) > xmin ...
                & ptCloud.Location(:,2) < ymax ...
                & ptCloud.Location(:,2) > ymin ...
                & ptCloud.Location(:,3) < zmax ...
                & ptCloud.Location(:,3) > zmin);    
            ptCloud = select(ptCloud, pos, 'OutputSize', 'full'); 
        end
        
        processedData = removeInvalidPoints(ptCloud);
         
        for ii = 1:numel(classNames)

            labels = groundTruth(1,classNames{ii}).Variables;
            if(iscell(labels))
                labels = labels{1};
            end
            if ~isempty(labels)

                % Get the label indices that are in the selected RoI.
                labelsIndices = labels(:,1) > xmin ...
                            & labels(:,1) < xmax ...
                            & labels(:,2) > ymin...
                            & labels(:,2) < ymax ...
                            & labels(:,4) > 0 ...
                            & labels(:,5) > 0 ...
                            & labels(:,6) > 0;
                labels = labels(labelsIndices,:);
             
                if ~isempty(labels)
                    % 如果过滤后标签不为空，进一步处理
                    % 查找每个标签中包含的点的数量。
                    numPoints = arrayfun(@(x)(findPointsInModel(cuboidModel(labels(x,:)),processedData)),...
                                (1:size(labels,1)).','UniformOutput',false);
                    posLabels = cellfun(@(x)(length(x) > 50), numPoints);       
                    labels = labels(posLabels,:);
                end
            end
            processedLabels{i,ii} = labels;
        end
        croppedPointCloudObj{i,1} = processedData;
    end
    
    % Print completion message when done.
    msg = sprintf('Processing data 100%% complete');
    fprintf(1,'%s',[tmpStr, msg]);

    processedLabels = cell2table(processedLabels);
    numClasses = size(processedLabels,2);
    for j = 1:numClasses
        processedLabels.Properties.VariableNames{j} = classNames{j};
    end

end