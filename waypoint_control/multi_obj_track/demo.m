%% 多目标跟踪的入口
%% 设置跟踪数据
config;
runFrameNum = 200;                             % 设置多目标跟踪帧数
juncNum = 1;                                   % 路口编号
junc = dataset.intersection_1;                 % 选择跟踪的路口
initTime = dataset.initialTime_1;              % 跟踪初始时间
transMatrix = dataset.TransformationMatrix_1;  % 转换矩阵
%% 获取2D检测框
detect2DBoundingBox(junc)
%% 获取点云3D检测框
detect3DBoundingBox(junc)
%% 在 Town10 路口1做多目标跟踪 
% 多目标跟踪生成轨迹并保存
multiObjectTracking(junc, initTime, runFrameNum);
% 将轨迹转换为Carla坐标并保存
convertTrackToCarlaCoordinate(junc, transMatrix);
%% 计算单个路口跟踪指标并保存
demoSingleJuncEvalution(junc, juncNum)