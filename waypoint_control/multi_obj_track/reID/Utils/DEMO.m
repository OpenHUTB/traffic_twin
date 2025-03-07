%% 将生成的轨迹与外观特征绑定
%% 匹配轨迹之前，处理单个路口的轨迹id变化的情况
config;
townName = 'Town10';   % 可以修改城镇和；路口
dataSets = {'Town10HD_Opt\test_data_junc1', 'Town10HD_Opt\test_data_junc2', 'Town10HD_Opt\test_data_junc3', 'Town10HD_Opt\test_data_junc4', 'Town10HD_Opt\test_data_junc5'};

dirParts = strsplit(dataSets{1}, '\');
townConfig = dataset.(townName);
for i = 1:length(dataSets)
    juncField = sprintf('intersection_%d', i);
    juncConfig = townConfig.(juncField);
    transMatrix = juncConfig.TransformationMatrix;
    loadAllTraj(dataSets{i}, transMatrix);
end
%% 加载所有轨迹
currentPath = fileparts(mfilename('fullpath'));
juncTracksFolderPath = fullfile(currentPath, dirParts{1});
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
matchThreshold = 0.65;  % 车辆匹配阈值
traj = linkIdentities(juncTrajCell, matchThreshold);

% 获取当前目录
currentFileDir = fileparts(mfilename('fullpath')); 

% 保存 traj 到当前目录下的 traj.mat 文件
save(fullfile(currentFileDir, 'traj.mat'), 'traj');
