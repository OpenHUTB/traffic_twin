%% 将生成的轨迹与外观特征绑定
%% 匹配轨迹之前，处理单个路口的轨迹id变化的情况
config;

data = load('/mnt/mydrive/traffic_twin/waypoint_control/multi_obj_track/matlab_final_timeline.mat');
t_start = data.t_end;
% 获取程序开始时间
sys_time_before = posixtime(datetime('now'));

townName = 'Town01';   % 可以修改城镇和路口
% dataSets = {'Town10HD_Opt/test_data_junc1', 'Town10HD_Opt/test_data_junc2', 'Town10HD_Opt/test_data_junc3', 'Town10HD_Opt/test_data_junc4', 'Town10HD_Opt/test_data_junc5'};
% 定义各城镇对应的数据集
if strcmp(townName, 'Town10')
    dataSets = {'Town10HD_Opt/test_data_junc1', ...
                'Town10HD_Opt/test_data_junc2', ...
                'Town10HD_Opt/test_data_junc3', ...
                'Town10HD_Opt/test_data_junc4', ...
                'Town10HD_Opt/test_data_junc5'};
elseif strcmp(townName, 'Town01')
    dataSets = {'Town01/test_data_junc1', ...
                'Town01/test_data_junc2', ...
                'Town01/test_data_junc3', ...
                'Town01/test_data_junc4', ...
                'Town01/test_data_junc5'};
else
    error('未知城镇: %s', townName);
end

dirParts = strsplit(dataSets{1}, filesep);
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
% 获取车辆轨迹文件
vehiclematFiles = dir(fullfile(juncTracksFolderPath, "*_vehicle_traj.mat"));
vehiclenumMatFiles = length(vehiclematFiles);
% 创建cell数组保存每个路口的轨迹
vehiclejuncTrajCell = cell(1,vehiclenumMatFiles);
for i = 1:vehiclenumMatFiles
   fileName = fullfile(juncTracksFolderPath, vehiclematFiles(i).name);
   data = load(fileName);
   vehiclejuncTrajCell{i} = data.juncVehicleTraj;
end

% 获取行人轨迹文件
personmatFiles = dir(fullfile(juncTracksFolderPath, "*_person_traj.mat"));
personnumMatFiles = length(personmatFiles);
% 创建cell数组保存每个路口的轨迹
personjuncTrajCell = cell(1,personnumMatFiles);
for i = 1:personnumMatFiles
   fileName = fullfile(juncTracksFolderPath, personmatFiles(i).name);
   data = load(fileName);
   personjuncTrajCell{i} = data.juncPersonTraj;
end

%% 轨迹匹配，链接全部路口的轨迹
% 车辆轨迹匹配
vehiclematchThreshold = 0.65;  % 车辆匹配阈值
vehicletraj = linkIdentities(vehiclejuncTrajCell, vehiclematchThreshold);

% 行人轨迹匹配
personmatchThreshold = 0.65;  % 车辆匹配阈值
persontraj = linkIdentities(personjuncTrajCell, personmatchThreshold);

% % 获取程序结束时间
% sys_time_after = posixtime(datetime('now'));
% % 计算程序耗时
% program_run_time = sys_time_after - sys_time_before;
% % 得出在线结束时间
% t_end = t_start + program_run_time;
% % 保存为.mat 文件
% save_folder = '/mnt/mydrive/traffic_twin/waypoint_control/multi_obj_track';
% txt_save_path = fullfile(save_folder, 'matlab_final.txt');
% fileID = fopen(txt_save_path, 'w');
% fprintf(fileID, '起始时间：%.6f\n终止时间：%.6f\n总耗时：%.4f', t_start, t_end, program_run_time);
% fclose(fileID);
% 
% 获取当前目录
currentFileDir = fileparts(mfilename('fullpath'));
savepath = fullfile(currentFileDir, townName);

% 分别保存轨迹文件
save(fullfile(savepath, 'vehicletraj.mat'), 'vehicletraj');
save(fullfile(savepath, 'persontraj.mat'), 'persontraj');
