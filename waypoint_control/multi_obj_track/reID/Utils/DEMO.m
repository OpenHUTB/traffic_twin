%% 将生成的轨迹与外观特征绑定
%% 匹配轨迹之前，处理单个路口的轨迹id变化的情况
config;
townNameFull = 'Town10HD_Opt';   % 可以修改城镇和路口

if strcmp(townName, 'Town10HD_Opt')
   townName = erase(townName, "HD_Opt");
end
%% 获取跟踪轨迹路径
currentPath = fileparts(mfilename('fullpath'));
% 获取当前路径的上级目录
parentPath = fileparts(currentPath);
% 再次获取上级目录，即上上级目录
grandparentPath = fileparts(parentPath);
addpath(grandparentPath)
dataPath = fullfile(grandparentPath, townNameFull);

%% 获取路口数量
d = dir(dataPath);
% 筛选出文件夹（排除 . 和 ..）
isDir = [d.isdir] & ~ismember({d.name}, {'.','..'});
% 文件夹数量
numFolders = sum(isDir);


dataSets = arrayfun(@(i) fullfile(dataPath, sprintf('test_data_junc%d', i)), ...
                    1:numFolders, 'UniformOutput', false);
dirParts = strsplit(dataSets{1}, '\');
townConfig = dataset.(townName);

for i = 1:length(dataSets)
    [p, last1, ~] = fileparts(dataSets{i});          % 'test_data_junc1'

    [p, last2, ~] = fileparts(p);                    % 'Town10HD_Opt'
    % 合并
    lastTwo = fullfile(last2, last1);                % 'Town10HD_Opt\test_data_junc1'

    juncField = sprintf('intersection_%d', i);
    juncConfig = townConfig.(juncField);
    transMatrix = juncConfig.TransformationMatrix;
    loadAllTraj(lastTwo, transMatrix);
end

%% 加载所有轨迹
currentPath = fileparts(mfilename('fullpath'));
juncTracksFolderPath = fullfile(currentPath, townNameFull);
% 获取所有轨迹文件
matFiles = dir(fullfile(juncTracksFolderPath, "*.mat"));
numMatFiles = length(matFiles);
% 创建cell数组保存每个路口的轨迹
juncTrajCell = cell(1,numMatFiles);
for file = 1:numMatFiles
   fileName = fullfile(juncTracksFolderPath, matFiles(file).name);
   data = load(fileName);
   juncTrajCell{file} = data.juncVehicleTraj;
end

%% 轨迹匹配，链接全部路口的轨迹
matchThreshold = 0.7;  % 车辆匹配阈值
traj = linkIdentities(juncTrajCell, matchThreshold);

% 获取当前目录
currentFileDir = fileparts(mfilename('fullpath')); 

% 保存 traj 到当前目录下的 traj.mat 文件
save(fullfile(currentFileDir, 'traj.mat'), 'traj');

computeSpeedCorrelations(townName)

computeDelay(townName)