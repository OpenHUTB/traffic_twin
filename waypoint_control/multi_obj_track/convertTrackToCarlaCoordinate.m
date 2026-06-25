function convertTrackToCarlaCoordinate(junc, TransformationMatrix)
    % 将融合的轨迹转换为carla中的坐标轨迹
    config;
    % 数据路径
    currentPath = fileparts(mfilename('fullpath'));
    dataPath = fullfile(currentPath, junc);
    vehiclefilename = fullfile(dataPath,'tracks',"trackedVehicleData.mat");
    personfilename = fullfile(dataPath,'tracks',"trackedPersonData.mat");
    % "dataLog": 这个参数指定加载文件中的变量 
    vehicledata = load(vehiclefilename);
    persondata = load(personfilename);
    vehicleTracks = vehicledata.vehicleTracks;
    personTracks = persondata.personTracks;

    % 遍历所有车辆轨迹
    for t = 1:length(vehicleTracks)
        % 获取当前轨迹的位置
        positions = vehicleTracks(t).Positions;  % [x, y, z] 在雷达坐标系中的位置矩阵
    
        % 对每个位置进行转换
        worldPositions = [];
        for i = 1:size(positions, 1)  % 遍历所有位置点
            radarPosition = [positions(i, :)'; 1];  % 将位置转换为齐次坐标 (x, y, z, 1)
            
            % 使用转换矩阵将雷达坐标系中的位置转换为 CARLA 世界坐标系
            worldPosition = TransformationMatrix * radarPosition;
            
            % 将转换后的世界坐标加入到 worldPositions 数组中
            worldPositions = [worldPositions; worldPosition(1:3)'];  % 取 x, y, z
        end
        
        % 更新轨迹中的位置为世界坐标
        vehicleTracks(t).Positions = worldPositions;
    end
    % 遍历所有行人轨迹
    for t = 1:length(personTracks)
        % 获取当前轨迹的位置
        positions = personTracks(t).Positions;  % [x, y, z] 在雷达坐标系中的位置矩阵
    
        % 对每个位置进行转换
        worldPositions = [];
        for i = 1:size(positions, 1)  % 遍历所有位置点
            radarPosition = [positions(i, :)'; 1];  % 将位置转换为齐次坐标 (x, y, z, 1)
            
            % 使用转换矩阵将雷达坐标系中的位置转换为 CARLA 世界坐标系
            worldPosition = TransformationMatrix * radarPosition;
            
            % 将转换后的世界坐标加入到 worldPositions 数组中
            worldPositions = [worldPositions; worldPosition(1:3)'];  % 取 x, y, z
        end
        
        % 更新轨迹中的位置为世界坐标
        personTracks(t).Positions = worldPositions;
    end
    % 轨迹目录
    tracksDirectory = fullfile(dataPath, "tracks");
    if ~exist(tracksDirectory, 'dir')
        mkdir(tracksDirectory);
    end
    % 保存转换后的轨迹
    vehiclesavePath = fullfile(tracksDirectory, 'vehicle_convertedCarlaTrackedData.mat');
    personsavePath = fullfile(tracksDirectory, 'person_convertedCarlaTrackedData.mat');
    save(vehiclesavePath, 'vehicleTracks');
    save(personsavePath, 'personTracks');
    disp(['转换后的车辆轨迹数据已保存到 ', vehiclesavePath]);
    disp(['转换后的行人轨迹数据已保存到 ', personsavePath]);
end