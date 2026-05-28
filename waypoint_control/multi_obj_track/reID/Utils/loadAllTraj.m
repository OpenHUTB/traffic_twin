function loadAllTraj(junc, transMatrix)
    config;
    currentPath = fileparts(mfilename('fullpath'));
    parentPath = fileparts(currentPath);
    grandparentPath = fileparts(parentPath);
    addpath(grandparentPath)
    
    dataPath = fullfile(grandparentPath, junc, 'tracks');
    trackedDataPath = fullfile(dataPath, 'trackedData.mat');
    
    % 加载跟踪数据
    if exist(trackedDataPath, 'file')
        loadedData = load(trackedDataPath);
    else
        disp('tracks does not exist');
        return;
    end
    
    % 获取轨迹总数
    numTraj = length(loadedData.allTracks);
    if numTraj == 0
        disp('No tracks found.');
        return;
    end
    
    traj_data = cell(1, numTraj);
    traj_f_data = zeros(numTraj, 2);
    
    trackerOutput.traj = traj_data;
    trackerOutput.traj_f = traj_f_data;
    
    % 遍历所有轨迹
    for i = 1:numTraj
        trackID = loadedData.allTracks(i).TrackID;
        positions = loadedData.allTracks(i).Positions;
        timestamps = loadedData.allTracks(i).Timestamps;
        
        % 提取特征和类别
        features = loadedData.allTracks(i).RepresentativeFeature;      %转为行向量
        features = features(:)';                          % 确保是 1xN 的行向量
        category = loadedData.allTracks(i).Categories;
        
        % 将雷达坐标系位置转换为世界坐标系
        worldPositions = [];
        for p = 1:size(positions, 1)
            radarPos = [positions(p, :)'; 1];
            worldPos = transMatrix * radarPos;
            worldPositions = [worldPositions; worldPos(1:3)'];
        end
        
        % 存入结构体
        trackerOutput.traj{i} = struct( ...
            'trackID', trackID, ...
            'wrl_pos', worldPositions, ...
            'mean_hsv', features, ...    
            'timestamp', timestamps, ...
            'category', category);
        
        trackerOutput.traj_f(i, :) = [timestamps(1), timestamps(end)];
    end
    
    % 后处理与保存
    juncVehicleTraj = processSingleJuncTraj(trackerOutput);
    
    baseName = 'traj';
    dirParts = strsplit(junc, filesep);
    fileName = [dirParts{2}, '_', baseName, '.mat'];
    juncTracksFolderPath = fullfile(currentPath, dirParts{1});
    if ~exist(juncTracksFolderPath, 'dir')
        mkdir(juncTracksFolderPath);
    end
    
    outputFilePath = fullfile(juncTracksFolderPath, fileName);
    save(outputFilePath, 'juncVehicleTraj');
    
    successMessage = [num2str(junc), ': trackerOutput saved successfully'];
    disp(successMessage);
end