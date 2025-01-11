%% 将融合的轨迹转换为carla中的坐标轨迹
% 数据路径
currentPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(currentPath, 'test_data');
filename = fullfile(dataPath,"trackedData.mat");
% "dataLog": 这个参数指定加载文件中的变量 
data = load(filename);
allTracks = data.allTracks;
%{
% 路口1：雷达到 carla 的转换矩阵
TransformationMatrix = [
    -4.37113883e-08, -1.00000000e+00,  0.00000000e+00, -4.60000000e+01;
     1.00000000e+00, -4.37113883e-08, -0.00000000e+00,  2.10000000e+01;
     0.00000000e+00, -0.00000000e+00,  1.00000000e+00,  1.79999995e+00;
     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00
];
%}
% 路口2：雷达到 carla 的转换矩阵
TransformationMatrix = [
     -4.37113883e-08, -1.00000000e+00, 0.00000000e+00, 1.04000000e+02;
     1.00000000e+00, -4.37113883e-08, -0.00000000e+00, 2.10000000e+01;
     0.00000000e+00, -0.00000000e+00, 1.00000000e+00, 1.79999995e+00;
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00
];
% 遍历所有轨迹
for t = 1:length(allTracks)
    % 获取当前轨迹的位置
    positions = allTracks(t).Positions;  % [x, y, z] 在雷达坐标系中的位置矩阵

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
    allTracks(t).Positions = worldPositions;
end

% 保存转换后的轨迹
savePath = fullfile(dataPath, 'convertedCarlaTrackedData.mat'); 
save(savePath, 'allTracks');
disp(['转换后的轨迹数据已保存到 ', savePath]);
