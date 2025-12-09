%% 多目标跟踪的入口
%% 设置跟踪数据
config;

% 用户输入地图名和路口编号
townName = 'Town10';                 % 例如 Town01 或 Town10）
juncNum = 1;                         % 请输入路口编号（1 或 2）

% 根据输入选择配置
if isfield(dataset, townName)
    townConfig = dataset.(townName);
    juncField = sprintf('intersection_%d', juncNum);
    if isfield(townConfig, juncField)
        juncConfig = townConfig.(juncField);
        
        % 设置跟踪参数
        runFrameNum = 500;                             % 设置多目标跟踪帧数
        junc = juncConfig.name;                        % 选择跟踪的路口
        initTime = juncConfig.initialTime;             % 跟踪初始时间
        transMatrix = juncConfig.TransformationMatrix; % 转换矩阵
        
        %% 获取2D检测框
        % detect2DBoundingBox(junc);
        
        %% 获取点云3D检测框
        % detect3DBoundingBox(junc, juncNum);
        
        %% 在指定路口做多目标跟踪 
        % 多目标跟踪生成轨迹并保存
        multiObjectTracking(junc, initTime, runFrameNum);
        
        % 将轨迹转换为Carla坐标并保存
        convertTrackToCarlaCoordinate(junc, transMatrix);
        
        %% 计算单个路口跟踪指标并保存
        %currentPath = fileparts(mfilename('fullpath'));
        %folderPath = fullfile(currentPath, 'Evaluation');
        % 将文件夹添加到路径
        %addpath(folderPath);
        %demoSingleJuncEvaluation(junc, juncConfig.juncNum);
        %if strcmp(townName, 'Town10')
        %    townPath = [townName 'HD_Opt_Metric'];
        %else
        %    townPath = [townName '_Metric'];
        %end
        %MCTPResults = demoMultiSensorEvaluation(townPath);
    else
        error('路口编号不存在！');
    end
else
    error('地图名不存在！');
end