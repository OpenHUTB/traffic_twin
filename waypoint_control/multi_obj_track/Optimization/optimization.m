%% 优化跟踪器的超参数 (多路口自动批量运行)
%% 设置跟踪数据
config;

% 设置需要优化的地图名和路口范围
townName = 'Town10';                 % 例如 Town01 或 Town10
juncList = 1:5;                      % 要优化的路口编号列表 (1到5号路口)
runFrameNum = 500;                   % 设置多目标跟踪帧数

% 首先检查地图名是否存在
if ~isfield(dataset, townName)
    error('地图名 %s 不存在！', townName);
end
townConfig = dataset.(townName);

% 开启并行池
if isempty(gcp('nocreate'))
    parpool; 
end

% 遍历要优化的所有路口
for i = 1:length(juncList)
    juncNum = juncList(i);
    
    fprintf('开始优化地图 %s 的 %d 号路口...\n', townName, juncNum);
    
    juncField = sprintf('intersection_%d', juncNum);
    
    % 检查该路口是否存在
    if isfield(townConfig, juncField)
        juncConfig = townConfig.(juncField);
        
        % 设置当前路口的跟踪参数
        junc = juncConfig.name;                        % 选择跟踪的路口
        initTime = juncConfig.initialTime;             % 跟踪初始时间
        
        %% 定义优化变量
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
        
        % 定义目标函数
        objectiveFcn = @(params)evaluateTracker(params, junc, initTime, runFrameNum);
        
        % 调用 bayesopt 进行优化
        results = bayesopt(objectiveFcn, vars, 'MaxObjectiveEvaluations', 100, ...
            'Verbose', 1, 'UseParallel', true);
        
        % 显示最优参数
        bestParams = results.XAtMinObjective;
        fprintf('%d 号路口最优参数：\n', juncNum);
        disp(bestParams);
        
        % 保存最优参数
        saveFileName = sprintf('OptResults_%s_Junc%d_%s.mat', townName, juncNum, datestr(now, 'yyyymmdd_HHMMSS'));
        save(saveFileName, 'results', 'bestParams', 'townName', 'juncNum');
        fprintf(' %d 号路口优化结果已成功保存至: %s\n', juncNum, saveFileName);
        
    else
        % 如果路口不存在，抛出警告而不是错误，确保后续路口继续运行
        warning('路口编号 %d (%s) 不存在，已跳过。', juncNum, juncField);
    end
end

fprintf('所有指定路口优化任务已全部完成！\n');
