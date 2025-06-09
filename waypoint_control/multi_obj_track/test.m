data = load('00000001.mat');  % 加载.mat文件

%pointCloud = datalog.LidarData.PointCloud;  % 假设变量名是pointCloud
pointCloud = data.sampledPtCld;



%% 2. 显示点云基本信息
disp('=== 点云基本信息 ===');
disp(['点数: ', num2str(pointCloud.Count)]);
disp(['X范围: [', num2str(pointCloud.XLimits(1)), ', ', num2str(pointCloud.XLimits(2)), ']']);
disp(['Y范围: [', num2str(pointCloud.YLimits(1)), ', ', num2str(pointCloud.YLimits(2)), ']']);
disp(['Z范围: [', num2str(pointCloud.ZLimits(1)), ', ', num2str(pointCloud.ZLimits(2)), ']']);

%% 3. 基础可视化（XYZ坐标）
figure;
pcshow(pointCloud.Location);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('原始点云（XYZ坐标）');
axis equal;  % 保持坐标轴比例一致


