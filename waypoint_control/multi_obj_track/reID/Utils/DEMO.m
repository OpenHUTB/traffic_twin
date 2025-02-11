% 初始化和配置
clc; close all; clear
% 添加一个工具函数文件夹路径
addpath(genpath('utils')); config; mydir = pwd;
cd('D:\utilities\gurobi\win64\matlab'); gurobi_setup; cd(mydir);

% 参数配置
dataset.frame_range = [56001 61000]; % ~5mins
dataset.t_window    = 5000; % ~40secs
dataset.group_size  = 80;
dataset.visualize   = false; % 设置为 false，表示暂时不进行可视化。
% 如果为 true，则重新计算目标的特征；如果为 false，则加载已经计算好的特征。
computeFeatures     = false; 

%% 加载轨迹并计算特征
if computeFeatures, [traj, traj_f] = loadAllTraj(dataset); save('traj.mat', 'traj', 'traj_f'); else load('traj.mat'); end %#ok

%% 计算身份
startTime = dataset.frame_range(1); endTime = dataset.frame_range(1) + dataset.t_window - 1;    % initialize range

while startTime <= dataset.frame_range(2)
    % print loop state
    clc; fprintf('Window %d...%d\n', startTime, endTime);
    
    % attach tracklets and store trajectories as they finish
    traj = linkIdentities(traj , traj_f, startTime, endTime, dataset);
    
    % update loop range
    startTime = endTime   - dataset.t_window/2;
    endTime   = startTime + dataset.t_window;
end
%  解析输出并保存
output = parseOutput(traj, dataset.cameras);
dlmwrite('final_output.txt', output, 'delimiter', ' ', 'precision', 6);

%% 可视化结果
drawOnMap(traj);