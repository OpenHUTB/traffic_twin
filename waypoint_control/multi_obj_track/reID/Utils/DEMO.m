%% 将生成的轨迹与外观特征绑定
loadAllTraj('test_data_junc1'); 
loadAllTraj('test_data_junc2');
%% 加载轨迹
junc1Tracks = load('test_data_junc1_traj.mat'); 
junc2Tracks = load('test_data_junc2_traj.mat'); 
%% 轨迹匹配
traj = linkIdentities(junc1Tracks.trackerOutput, junc2Tracks.trackerOutput);

% 获取当前目录
currentFileDir = fileparts(mfilename('fullpath')); 

% 保存 traj 到当前目录下的 traj.mat 文件
save(fullfile(currentFileDir, 'traj.mat'), 'traj');
%% 可视化结果
%drawOnMap(traj);