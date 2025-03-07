function loadAllTraj(junc, transMatrix)
    config;
    currentPath = fileparts(mfilename('fullpath'));
    % 获取当前路径的上级目录
    parentPath = fileparts(currentPath);
    % 再次获取上级目录，即上上级目录
    grandparentPath = fileparts(parentPath);
    addpath(grandparentPath)
    dataPath = fullfile(grandparentPath, junc,'tracks' );
    trackedDataPath = fullfile(dataPath, 'trackedData.mat');
    datasetFolder = "trainedCustomReidNetwork.mat";
    netFolder = fullfile(grandparentPath, datasetFolder);
    data = load(netFolder);
    net = data.net;
    % 加载 .mat 文件中的数据
    if exist(trackedDataPath, 'file')
        loadedData = load(trackedDataPath);  % 加载文件内容
    else
        disp('tracks does not exist');
    end
    
    tracksVehiclePicturePath = fullfile(parentPath, 'trkIDImg', junc);
    
    % 获取目录下所有文件
    imageFiles = dir(fullfile(tracksVehiclePicturePath, '*.jpeg')); % 或者 '*.jpeg', '*.png' 根据你的图片格式调整
    numImages = numel(imageFiles);
    
    traj_data = cell(1, numImages); 
    traj_f_data = zeros(numImages, 2); 
    
    % 将数据保存到结构体中
    trackerOutput.traj = traj_data;
    trackerOutput.traj_f = traj_f_data;
    % 遍历每张图片
    for k = 1:length(imageFiles)
        % 获取当前图片的完整路径
        imageFilePath = fullfile(tracksVehiclePicturePath, imageFiles(k).name);
         % 加载图片
        img = imread(imageFilePath);
        % 从文件名中提取轨迹ID
        [~, imageName, ~] = fileparts(imageFiles(k).name);  % 提取文件名，不带扩展名
        trackID = str2double(imageName);  % 将文件名转换为数字作为轨迹ID
        % 在表格中找到对应的行
        index = find([loadedData.allTracks.TrackID] == trackID);
        if isempty(index)
            continue;  % 跳过当前循环，继续下一个
        end
        positions = loadedData.allTracks(index).Positions;
        % 将位置转换成Carla中的三维坐标
        worldPositions = [];
        for i = 1:size(positions, 1)  % 遍历所有位置点
            radarPosition = [positions(i, :)'; 1];  % 将位置转换为齐次坐标 (x, y, z, 1)
            % 使用转换矩阵将雷达坐标系中的位置转换为 CARLA 世界坐标系
            worldPosition = transMatrix * radarPosition;
            % 将转换后的世界坐标加入到 worldPositions 数组中
            worldPositions = [worldPositions; worldPosition(1:3)'];  % 取 x, y, z
        end
       
        features = zeros(2048,1);
        features(:,1) = extractReidentificationFeatures(net,img);
        % 将特征重塑为 1x2048 的形式
        features = reshape(features, 1, 2048);
        timestamp = loadedData.allTracks(index).Timestamps;
        trackerOutput.traj{k} = struct( ...
            'trackID', trackID, ...    % 轨迹 ID
            'wrl_pos', worldPositions, ...  % 位置数据
            'mean_hsv', features, ...  % 特征数据
            'timestamp', timestamp ... % 轨迹时间
        );
       trackerOutput.traj_f(k,:) = [timestamp(1), timestamp(end)];
    end
    juncVehicleTraj = processSingleJuncTraj(trackerOutput);
    baseName = 'traj';
    dirParts = strsplit(junc, '\');
    fileName = [dirParts{2}, '_', baseName, '.mat'];
    juncTracksFolderPath = fullfile(currentPath, dirParts{1});
    if ~exist(juncTracksFolderPath, 'dir')
        mkdir(juncTracksFolderPath);
    end
    % 保存 trackerOutput 到 .mat 文件
    outputFilePath = fullfile(juncTracksFolderPath, fileName);  % 定义保存路径
    save(outputFilePath, 'juncVehicleTraj');  
    successMessage = [num2str(junc), ': trackerOutput saved successfully' ];
    disp(successMessage);
end 