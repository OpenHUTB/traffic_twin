function loadAllTraj(junc, transMatrix)
    config;
    currentPath = fileparts(mfilename('fullpath'));
    parentPath = fileparts(currentPath);
    grandparentPath = fileparts(parentPath);
    addpath(grandparentPath)
    
    % 定义数据路径
    dataPath = fullfile(grandparentPath, junc, 'tracks');
    vehicleDataPath = fullfile(dataPath, 'trackedVehicleData.mat');
    personDataPath = fullfile(dataPath, 'trackedPersonData.mat');
    
    % 准备输出文件夹
    dirParts = strsplit(junc, filesep);
    juncTracksFolderPath = fullfile(currentPath, dirParts{1});
    if ~exist(juncTracksFolderPath, 'dir')
        mkdir(juncTracksFolderPath);
    end

    %% 处理并保存车辆数据 
    if exist(vehicleDataPath, 'file')
        vData = load(vehicleDataPath);
        % 动态获取 .mat 文件里的变量 
        fields = fieldnames(vData);
        vTracks = vData.(fields{1}); 
        
        if isempty(vTracks)
            disp([junc, ': No vehicle tracks found.']);
        else
            trackerOutputV = buildTrackerOutput(vTracks, transMatrix);
            juncVehicleTraj = processSingleJuncTraj(trackerOutputV);
            
            % 保存车辆轨迹
            fileNameV = [dirParts{2}, '_vehicle_traj.mat'];
            outputFilePathV = fullfile(juncTracksFolderPath, fileNameV);
            save(outputFilePathV, 'juncVehicleTraj');
            disp([junc, ': Vehicle trackerOutput saved successfully']);
        end
    else
        disp([junc, ': Vehicle tracks does not exist']);
    end
    
    %% 处理并保存行人数据
    if exist(personDataPath, 'file')
        pData = load(personDataPath);
        fields = fieldnames(pData);
        pTracks = pData.(fields{1});
        
        if isempty(pTracks)
            disp([junc, ': No person tracks found.']);
        else
            trackerOutputP = buildTrackerOutput(pTracks, transMatrix);
            juncPersonTraj = processSingleJuncTraj(trackerOutputP);
            
            % 保存行人轨迹
            fileNameP = [dirParts{2}, '_person_traj.mat'];
            outputFilePathP = fullfile(juncTracksFolderPath, fileNameP);
            save(outputFilePathP, 'juncPersonTraj');
            disp([junc, ': Person trackerOutput saved successfully']);
        end
    else
        disp([junc, ': Person tracks does not exist']);
    end
end

%% 辅助函数 
function trackerOutput = buildTrackerOutput(tracks, transMatrix)
    numTraj = length(tracks);
    
    traj_data = cell(1, numTraj);
    traj_f_data = zeros(numTraj, 2);
    
    trackerOutput.traj = traj_data;
    trackerOutput.traj_f = traj_f_data;
    
    for i = 1:numTraj
        trackID = tracks(i).TrackID;
        positions = tracks(i).Positions;
        timestamps = tracks(i).Timestamps;
        
        % 提取特征和类别
        features = tracks(i).RepresentativeFeature;      
        features = features(:)';                          % 确保是 1xN 的行向量
        category = tracks(i).Categories;
        
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
end