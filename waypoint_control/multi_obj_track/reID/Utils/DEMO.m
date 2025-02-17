junc1Tracks = load('test_data_junc1_traj.mat'); 
junc2Tracks = load('test_data_junc2_traj.mat'); 
traj = linkIdentities(junc1Tracks.trackerOutput, junc2Tracks.trackerOutput);

%% 可视化结果
%drawOnMap(traj);