% 加载两个 .mat 文件
data1 = load('Town10HD_Opt/pedestrian_ground_truth.mat');
data2 = load('Town10HD_Opt/vehicle_ground_truth.mat');

% 横向合并（水平连接）
truth_data = [data1.pedestrian_cells, data2.vehicle_cells];

% 保存合并后的数据到新文件
save('ground_truth.mat', 'truth_data');