%% 将生成的轨迹与外观特征绑定
%% 匹配轨迹之前，处理单个路口的轨迹id变化的情况
dataSets = {'Town10HD_Opt\test_data_junc1', 'Town10HD_Opt\test_data_junc2', 'Town10HD_Opt\test_data_junc3', 'Town10HD_Opt\test_data_junc4', 'Town10HD_Opt\test_data_junc5'};
for i = 1:length(dataSets)
    loadAllTraj(dataSets{i});
end
%% 加载所有轨迹
currentPath = fileparts(mfilename('fullpath'));
dirParts = strsplit(dataSets{1}, '\');
juncTracksFolderPath = fullfile(currentPath, dirParts{1});
% 获取所有轨迹文件
matFiles = dir(fullfile(juncTracksFolderPath, "*.mat"));
numMatFiles = length(matFiles);
for file = 1:numMatFiles
   fileName = fullfile(juncTracksFolderPath, matFiles(file).name);
end

%% 轨迹匹配，链接全部路口的轨迹
traj = linkIdentities(junc1Tracks.trackerOutput, junc2Tracks.trackerOutput);

% 获取当前目录
currentFileDir = fileparts(mfilename('fullpath')); 

% 保存 traj 到当前目录下的 traj.mat 文件
save(fullfile(currentFileDir, 'traj.mat'), 'traj');
%% 可视化结果
%drawOnMap(traj);