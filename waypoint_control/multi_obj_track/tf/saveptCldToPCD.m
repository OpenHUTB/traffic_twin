function [ptCld, ptLabels] = saveptCldToPCD(ptCld, ptLabels, dataLocation, writeFiles)
% This function saves the required point clouds in the specified location 

    if ~exist(dataLocation, 'dir')
        mkdir(dataLocation)
    end
    
    tmpStr = '';
    numFiles = size(ptLabels,1);
    ind = [];
    
    for i = 1:numFiles
        processedData = ptCld{i,1};
        
        % Skip if the processed point cloud is empty
        if(isempty(processedData.Location))
            ind = [ind, i];
            continue;
        end
        
        if(writeFiles)
            pcFilePath = fullfile(dataLocation, sprintf('%06d.pcd',i));
            pcwrite(processedData, pcFilePath);
        end
      
        % Display progress after 300 files on screen.
        if ~mod(i,300)
            msg = sprintf('Processing data %3.2f%% complete', (i/numFiles)*100.0);
            fprintf(1,'%s',[tmpStr, msg]);
            tmpStr = repmat(sprintf('\b'), 1, length(msg));
        end
    end
    
    ptCld(ind,:) = [];
    ptLabels(ind,:) = [];
    
    % Print completion message when done.
    msg = sprintf('Processing data 100%% complete');
    fprintf(1,'%s',[tmpStr, msg]);
end

        