%% 转换训练点云mat为重组pcd格式

% 获取当前文件路径和数据文件夹路径
currentPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(currentPath, 'train_data');

% 确保目标文件夹存在
pcdPath = fullfile(currentPath, 'Lidar');
if ~exist(pcdPath, 'dir')
    mkdir(pcdPath); % 创建文件夹
end

% 获取 dataPath 下所有 .mat 文件
matFiles = dir(fullfile(dataPath, '*.mat'));

% 遍历每个 .mat 文件
for i = 1:length(matFiles)
    % 获取当前文件名
    matFileName = fullfile(dataPath, matFiles(i).name);
    
    % 加载 .mat 文件
    data = load(matFileName);
    datalog = data.datalog;
    
    % 提取点云数据
    points = datalog.LidarData.PointCloud.Location;
    intensity = datalog.LidarData.PointCloud.Intensity;
    ptCloud = pointCloud(points, 'Intensity', intensity);
    
    % 转换成有组织的点云序列
    horizontalResolution = 1024;
    params = lidarParameters('HDL64E', horizontalResolution);
    ptCloudOrd = pcorganize(ptCloud, params);
    
    % 定义输出 PCD 文件路径
    [~, name, ~] = fileparts(matFiles(i).name); % 获取文件名（不带扩展名）
    outputFilename = fullfile(pcdPath, [name, '.pcd']); % 定义 PCD 文件路径
    
    % 保存为 PCD 文件
    pcwrite(ptCloudOrd, outputFilename, 'Encoding', 'ascii'); % 保存为 ASCII 格式的 PCD 文件
    
    % 显示进度
    disp(['已转换文件: ', matFiles(i).name, ' --> ', outputFilename]);
end

disp('所有文件转换完成！');
