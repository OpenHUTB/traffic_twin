%% 优化跟踪器的超参数
%% 设置跟踪数据
config;

% 用户输入地图名和路口编号
townName = 'Town01';                 % 例如 Town01 或 Town10）
juncNum = 1;                         % 请输入路口编号（1,2,3,4,5）

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
        %% 在指定路口优化跟踪器
        vars = [
            optimizableVariable('DetectionProbability', [0.5, 0.9]), ...
            optimizableVariable('ClutterDensity', [1e-8, 1e-6], 'Transform', 'log'), ...
            optimizableVariable('NewTargetDensity', [1e-7, 1e-5], 'Transform', 'log'), ...
            optimizableVariable('ConfirmationThreshold', [0.8, 0.99]), ...
            optimizableVariable('DeletionThreshold', [0.2, 0.6]), ...
            optimizableVariable('DeathRate', [0.3, 0.7])
        ];
        objectiveFcn = @(params)evaluateTracker(params, junc, initTime, runFrameNum);
        % 调用 bayesopt 进行优化
        results = bayesopt(objectiveFcn, vars, 'MaxObjectiveEvaluations', 50, ...
            'Verbose', 1, 'UseParallel', false);

        % 显示最优参数
        bestParams = results.XAtMinObjective;
        disp('最优参数：');
        disp(bestParams);
    else
        error('路口编号不存在！');
    end
else
    error('地图名不存在！');
end