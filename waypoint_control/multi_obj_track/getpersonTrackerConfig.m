function params = getpersonTrackerConfig(mapName, juncName)
    % 定义所有路口的跟踪器的超参数
    
    switch mapName
        case 'Town01'
            switch juncName
                case 'test_data_junc1'
                    params.AssignmentThreshold   = [1.11739583095735 15.8814219908511];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.937595606689869;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 8.04249655499483e-06;
                    params.NewTargetDensity      = 1.09325783874401e-05;
                    params.ConfirmationThreshold = 0.903053308146516;
                    params.DeletionThreshold     = 0.230258687740478;
                    params.DeathRate             = 0.669498250122228;
            
                case 'test_data_junc2'
                    % 为路口2设置不同的参数
                    params.AssignmentThreshold   = [2.72230123533656 12.5460227030653];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.855289312694044;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 0.000659240337500979;
                    params.NewTargetDensity      = 0.000591231580808432;
                    params.ConfirmationThreshold = 0.940741177165994;
                    params.DeletionThreshold     = 0.297490584524257;
                    params.DeathRate             = 0.538205842603255;
            
                case 'test_data_junc3'
                    % 为路口3设置不同的参数
                    params.AssignmentThreshold   = [5.87125909919786 13.5728780546997];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.873117008982223;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 3.07835010027278e-05;
                    params.NewTargetDensity      = 0.000121908770381848;
                    params.ConfirmationThreshold = 0.930240067544257;
                    params.DeletionThreshold     = 0.341832017924167;
                    params.DeathRate             = 0.324585970621323;
            
                case 'test_data_junc4'
                    % 为路口4设置不同的参数
                    params.AssignmentThreshold   = [2.69528199852330 16.8774441871042];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.944717414070525;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 0.000993755299533813;
                    params.NewTargetDensity      = 0.00133720921785614;
                    params.ConfirmationThreshold = 0.630242583825310;
                    params.DeletionThreshold     = 0.348875601893628;
                    params.DeathRate             = 0.610313578638342;
            
                case 'test_data_junc5'
                    % 为路口5设置不同的参数
                    params.AssignmentThreshold   = [3.38369163501663 16.0807146788425];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.706273201335009;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 0.000791779468725156;
                    params.NewTargetDensity      = 0.00147091618224815;
                    params.ConfirmationThreshold = 0.945146473586896;
                    params.DeletionThreshold     = 0.487301452073911;
                    params.DeathRate             = 0.617210877411384;
            
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