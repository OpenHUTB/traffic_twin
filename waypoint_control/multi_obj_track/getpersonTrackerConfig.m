function params = getpersonTrackerConfig(juncName)
    % 定义所有路口的跟踪器的超参数
    % 通过贝叶斯优化所得
    
    switch juncName
        case 'test_data_junc1'
            params.AssignmentThreshold   = [3.95474239120096 15.9538018552425];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.585278673310508;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 0.000983360286192015;
            params.NewTargetDensity      = 0.00112332609659550;
            params.ConfirmationThreshold = 0.847521738320261;
            params.DeletionThreshold     = 0.203288530880363;
            params.DeathRate             = 0.616998616792471;
            
        case 'test_data_junc2'
            % 为路口2设置不同的参数
            params.AssignmentThreshold   = [8.466252929125072 10.3258395323113];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.946991662899689;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 0.000603575512544274;
            params.NewTargetDensity      = 0.000284770718893464;
            params.ConfirmationThreshold = 0.664834388195429;
            params.DeletionThreshold     = 0.293972520542182;
            params.DeathRate             = 0.308886871217613;
            
        case 'test_data_junc3'
            % 为路口3设置不同的参数
            params.AssignmentThreshold   = [5.46840649097910 12.0147643384456];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.948425757123005;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 6.08205948444001e-05;
            params.NewTargetDensity      = 3.20349625616238e-05;
            params.ConfirmationThreshold = 0.927170093641791;
            params.DeletionThreshold     = 0.270371447938259;
            params.DeathRate             = 0.497036702907582;
            
        case 'test_data_junc4'
            % 为路口4设置不同的参数
            params.AssignmentThreshold   = [5.09143609475520 14.6513605639033];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.940947140309850;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 0.000925689065965876;
            params.NewTargetDensity      = 0.000647926094117131;
            params.ConfirmationThreshold = 0.932979781645390;
            params.DeletionThreshold     = 0.324245224023571;
            params.DeathRate             = 0.589126950104935;
            
        case 'test_data_junc5'
            % 为路口5设置不同的参数
            params.AssignmentThreshold   = [1.02539975584865 10.6721331848200];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.849872138605751;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 0.000794584885039360;
            params.NewTargetDensity      = 0.000543015572109355;
            params.ConfirmationThreshold = 0.939932427989970;
            params.DeletionThreshold     = 0.284493087829871;
            params.DeathRate             = 0.541236893414176;
            
        otherwise
            error('未知的路口名称: %s. 请使用 junc1 到 junc5', juncName);
    end
end