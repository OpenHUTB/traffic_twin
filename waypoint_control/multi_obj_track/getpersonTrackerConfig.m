function params = getpersonTrackerConfig(mapName, juncName)
    % 定义所有路口的跟踪器的超参数
    
    switch mapName
        case 'Town01'
            switch juncName
                case 'test_data_junc1'
                    params.AssignmentThreshold   = [2.97747335296038 14.5524897833759];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.943321585537362;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 0.000866240568336769;
                    params.NewTargetDensity      = 0.000419662491532024;
                    params.ConfirmationThreshold = 0.901631900495953;
                    params.DeletionThreshold     = 0.291609699644504;
                    params.DeathRate             = 0.670290537394368;
            
                case 'test_data_junc2'
                    % 为路口2设置不同的参数
                    params.AssignmentThreshold   = [8.23672238875688 15.7654598129533];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.948756235152492;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 0.000936866119851987;
                    params.NewTargetDensity      = 0.00675185082800010;
                    params.ConfirmationThreshold = 0.943711977295654;
                    params.DeletionThreshold     = 0.589836499677124;
                    params.DeathRate             = 0.553835116154699;
            
                case 'test_data_junc3'
                    % 为路口3设置不同的参数
                    params.AssignmentThreshold   = [8.84091737458980 13.1604487499737];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.519428154683526;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 0.000949850659056131;
                    params.NewTargetDensity      = 1.00824390435394e-05;
                    params.ConfirmationThreshold = 0.862022498749934;
                    params.DeletionThreshold     = 0.490672384640826;
                    params.DeathRate             = 0.332379789131892;
            
                case 'test_data_junc4'
                    % 为路口4设置不同的参数
                    params.AssignmentThreshold   = [3.92851813360968 16.8866840594349];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.933675352705422;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 0.000480307134861586;
                    params.NewTargetDensity      = 0.000843651073425903;
                    params.ConfirmationThreshold = 0.948494738478409;
                    params.DeletionThreshold     = 0.574931226042544;
                    params.DeathRate             = 0.657485915153470;
            
                case 'test_data_junc5'
                    % 为路口5设置不同的参数
                    params.AssignmentThreshold   = [9.45764148519469 12.0410452947432];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.939206172741026;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 0.000896326081357801;
                    params.NewTargetDensity      = 0.000478085270621507;
                    params.ConfirmationThreshold = 0.947584565574986;
                    params.DeletionThreshold     = 0.220514737197920;
                    params.DeathRate             = 0.647795707963701;
            
                otherwise
                    error('未知的路口名称: %s. 请使用 junc1 到 junc5', juncName);
            end

        case 'Town10HD_Opt'
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
end