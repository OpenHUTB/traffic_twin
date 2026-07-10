function params = getvehicleTrackerConfig(mapName, juncName)
    % 定义所有路口的跟踪器的超参数

    switch mapName
        case 'Town01'
            switch juncName
                case 'test_data_junc1'
                    params.AssignmentThreshold   = [2.45388204828801 15.2985175725122];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.873481221682843;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 7.77423905834477e-07;
                    params.NewTargetDensity      = 1.08921011114548e-06;
                    params.ConfirmationThreshold = 0.864432285931773;
                    params.DeletionThreshold     = 0.392294713170270;
                    params.DeathRate             = 0.366475549361350;
            
                case 'test_data_junc2'
                    % 为路口2设置不同的参数
                    params.AssignmentThreshold   = [13.5709965483189 17.2460423697133];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.896613727728421;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 7.85736061175583e-08;
                    params.NewTargetDensity      = 1.44429789941436e-07;
                    params.ConfirmationThreshold = 0.949470277923141;
                    params.DeletionThreshold     = 0.365085821448857;
                    params.DeathRate             = 0.548734743828175;
            
                case 'test_data_junc3'
                    % 为路口3设置不同的参数
                    params.AssignmentThreshold   = [2.11462850714591 21.1101660634957];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.893897003235471;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 1.08187351723186e-07;
                    params.NewTargetDensity      = 1.71737532836353e-07;
                    params.ConfirmationThreshold = 0.898604790469875;
                    params.DeletionThreshold     = 0.586087844163779;
                    params.DeathRate             = 0.362059107137417;
            
                case 'test_data_junc4'
                    % 为路口4设置不同的参数
                    params.AssignmentThreshold   = [2.58887872709735 54.7516789284473];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.878952056917922;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 5.31965176074703e-07;
                    params.NewTargetDensity      = 6.07855459589331e-07;
                    params.ConfirmationThreshold = 0.983401587201201;
                    params.DeletionThreshold     = 0.428104981045802;
                    params.DeathRate             = 0.302375614069101;
            
                case 'test_data_junc5'
                    % 为路口5设置不同的参数
                    params.AssignmentThreshold   = [13.9386778508241 22.4121305704114];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.536682102839997;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 3.40153471399495e-07;
                    params.NewTargetDensity      = 1.00338737544349e-07;
                    params.ConfirmationThreshold = 0.830448044371134;
                    params.DeletionThreshold     = 0.573481117328615;
                    params.DeathRate             = 0.369064811153443;
            
                otherwise
                    error('未知的路口名称: %s. 请使用 junc1 到 junc5', juncName);
            end

        case 'Town10HD_Opt'
            switch juncName
                case 'test_data_junc1'
                    params.AssignmentThreshold   = [14.6979127046354 59.4320158790005];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.758093156059073;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 1.07427576047953e-07;
                    params.NewTargetDensity      = 1.62405306444881e-07;
                    params.ConfirmationThreshold = 0.893053619243481;
                    params.DeletionThreshold     = 0.477080378123755;
                    params.DeathRate             = 0.629968213778632;
            
                case 'test_data_junc2'
                    % 为路口2设置不同的参数
                    params.AssignmentThreshold   = [2.70917790011311 58.7909920491904];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.893444154077815;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 3.95521738360293e-07;
                    params.NewTargetDensity      = 5.56515700807898e-07;
                    params.ConfirmationThreshold = 0.985239270992192;
                    params.DeletionThreshold     = 0.536268471712125;
                    params.DeathRate             = 0.656012871917509;
            
                case 'test_data_junc3'
                    % 为路口3设置不同的参数
                    params.AssignmentThreshold   = [2.06330547100064 23.1584216661660];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.863209193340532;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 1.46425470467113e-07;
                    params.NewTargetDensity      = 7.17310143105499e-07;
                    params.ConfirmationThreshold = 0.953157284173338;
                    params.DeletionThreshold     = 0.507176124383167;
                    params.DeathRate             = 0.354818528953035;
            
                case 'test_data_junc4'
                    % 为路口4设置不同的参数
                    params.AssignmentThreshold   = [2.19489305159653 19.6894953874025];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.733420638716348;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 5.80498225895906e-08;
                    params.NewTargetDensity      = 1.78791519188703e-07;
                    params.ConfirmationThreshold = 0.891786563091118;
                    params.DeletionThreshold     = 0.530538681895739;
                    params.DeathRate             = 0.635792344151171;
            
                case 'test_data_junc5'
                    % 为路口5设置不同的参数
                    params.AssignmentThreshold   = [2.01303727420413 34.5997879316485];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.898126960859168;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 1.26988768554027e-08;
                    params.NewTargetDensity      = 1.11797792447531e-07;
                    params.ConfirmationThreshold = 0.898713815042262;
                    params.DeletionThreshold     = 0.506366285155928;
                    params.DeathRate             = 0.452428382735471;
            
                otherwise
                    error('未知的路口名称: %s. 请使用 junc1 到 junc5', juncName);
            end
        
    end
end