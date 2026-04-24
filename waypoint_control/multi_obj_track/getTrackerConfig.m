function params = getTrackerConfig(juncName)
    % 定义所有路口的跟踪器的超参数
    
    switch juncName
        case 'test_data_junc1'
            params.AssignmentThreshold   = [14.6544507369167 55.0457432966172];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.791861721138464;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 9.82634209806117e-07;
            params.NewTargetDensity      = 1.27817603112830e-06;
            params.ConfirmationThreshold = 0.932797716364672;
            params.DeletionThreshold     = 0.597837621090455;
            params.DeathRate             = 0.644995093626921;
            
        case 'test_data_junc2'
            % 为路口2设置不同的参数
            params.AssignmentThreshold   = [14.8925195665092 19.2039449626418];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.896601850501315;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 9.74575482125966e-07;
            params.NewTargetDensity      = 5.72708620415610e-07;
            params.ConfirmationThreshold = 0.865295203097718;
            params.DeletionThreshold     = 0.448539815013242;
            params.DeathRate             = 0.694726246938953;
            
        case 'test_data_junc3'
            % 为路口3设置不同的参数
            params.AssignmentThreshold   = [14.0145516165050 51.9655488006945];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.871675349912356;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 9.96066413422530e-07;
            params.NewTargetDensity      = 4.34974038620946e-07;
            params.ConfirmationThreshold = 0.922107965103109;
            params.DeletionThreshold     = 0.382264215065847;
            params.DeathRate             = 0.643192897084099;
            
        case 'test_data_junc4'
            % 为路口4设置不同的参数
            params.AssignmentThreshold   = [11.4792005179845 27.4987897044205];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.614096426070744;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 3.38197488920524e-07;
            params.NewTargetDensity      = 1.03024391805103e-07;
            params.ConfirmationThreshold = 0.917211802436127;
            params.DeletionThreshold     = 0.512275105680075;
            params.DeathRate             = 0.682482918207740;
            
        case 'test_data_junc5'
            % 为路口5设置不同的参数
            params.AssignmentThreshold   = [14.7410935768934 15.9711688341553];
            params.MaxNumTracks          = 500;
            params.DetectionProbability  = 0.871528605383015;
            params.MaxNumEvents          = 50;
            params.ClutterDensity        = 8.83174692900613e-07;
            params.NewTargetDensity      = 1.76937056491499e-07;
            params.ConfirmationThreshold = 0.845015468076006;
            params.DeletionThreshold     = 0.476328115951925;
            params.DeathRate             = 0.649154348255219;
            
        otherwise
            error('未知的路口名称: %s. 请使用 junc1 到 junc5', juncName);
    end
end