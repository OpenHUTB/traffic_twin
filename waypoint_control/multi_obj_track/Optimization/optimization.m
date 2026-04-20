%% 优化跟踪器的超参数
%% 设置跟踪数据
config;

% 用户输入地图名和路口编号
townName = 'Town10';                 % 例如 Town01 或 Town10）
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
            optimizableVariable('DeathRate', [0.3, 0.7]), ...
            optimizableVariable('AssignThresh1', [2, 15]), ...  % 粗门控阈值
            optimizableVariable('AssignThresh2', [15, 60])      % 精细门控阈值
        ];
        objectiveFcn = @(params)evaluateTracker(params, junc, initTime, runFrameNum);
        % 开启并行池
        if isempty(gcp('nocreate'))
            parpool; 
        end
        % 调用 bayesopt 进行优化
        results = bayesopt(objectiveFcn, vars, 'MaxObjectiveEvaluations', 100, ...
            'Verbose', 1, 'UseParallel', true);

        % 显示最优参数
        bestParams = results.XAtMinObjective;
        disp('最优参数：');
        disp(bestParams);
        % 保存最优参数
        saveFileName = sprintf('OptResults_%s_Junc%d_%s.mat', townName, juncNum, datestr(now, 'yyyymmdd_HHMMSS'));
        save(saveFileName, 'results', 'bestParams', 'townName', 'juncNum');
        fprintf('优化结果已保存至: %s\n', saveFileName);
    else
        error('路口编号不存在！');
    end
else
    error('地图名不存在！');
end