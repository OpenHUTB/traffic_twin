%% 转换点云标签为单个文件的table变量的CarlaSetLidarGroundTruth.mat

% 初始化空表格
LabelData = table();
% 获取当前文件路径和数据文件夹路径
currentPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(currentPath, 'train_data', 'label');

% 确保目标文件夹存在
labelPath = fullfile(currentPath, 'Cuboids');
if ~exist(labelPath, 'dir')
    mkdir(labelPath); % 创建文件夹
end

% 获取 dataPath 下所有 .mat 文件
matFiles = dir(fullfile(dataPath, '*.mat'));
% 遍历每个 .mat 文件
for i = 1:length(matFiles)
    matFileName = fullfile(dataPath, matFiles(i).name);

    % 加载 mat 文件
    fileData = load(matFileName);
    % 获取 car 和 truck 的行数

    carData = fileData.LabelData.car;
    truckData = fileData.LabelData.truck;
    % 将 carData 转换为双精度数组
    carDataDouble = cell2mat(carData);
    % 将 truckData 转换为双精度数组
    truckDataDouble = cell2mat(truckData);
    % 将双精度数组存储在元胞数组中
    carCell = {carDataDouble};
    truckCell = {truckDataDouble};
    % 创建一个表格行
    newRow = table(fileData.LabelData.Time, carCell, truckCell, ...
                       'VariableNames', {'Time', 'car', 'truck'});
    % 将新行添加到主表格
    LabelData = [LabelData; newRow]; 
end
save(fullfile(labelPath, 'CarlaSetLidarGroundTruth.mat'), 'LabelData');