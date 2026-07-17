function params = getvehicleTrackerConfig(mapName, juncName)
    % 定义所有路口的跟踪器的超参数

    switch mapName
        case 'Town01'
            switch juncName
                case 'test_data_junc1'
                    params.AssignmentThreshold   = [10.5931980769760 20.6109007224522];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.746037350223033;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 1.98707373973405e-08;
                    params.NewTargetDensity      = 1.38935371142489e-07;
                    params.ConfirmationThreshold = 0.988596149418680;
                    params.DeletionThreshold     = 0.551207894776409;
                    params.DeathRate             = 0.564283920861690;
            
                case 'test_data_junc2'
                    % 为路口2设置不同的参数
                    params.AssignmentThreshold   = [11.4964922403104 24.7043186375859];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.893460614566182;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 9.95052015370506e-07;
                    params.NewTargetDensity      = 6.31489337354542e-06;
                    params.ConfirmationThreshold = 0.983604870623160;
                    params.DeletionThreshold     = 0.516303809270453;
                    params.DeathRate             = 0.324570732835291;
            
                case 'test_data_junc3'
                    % 为路口3设置不同的参数
                    params.AssignmentThreshold   = [3.41774326286932 29.9718149294276];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.831434037698859;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 9.98759128985741e-07;
                    params.NewTargetDensity      = 1.92305080215396e-06;
                    params.ConfirmationThreshold = 0.866501524619089;
                    params.DeletionThreshold     = 0.515993052659001;
                    params.DeathRate             = 0.374056886065844;
            
                case 'test_data_junc4'
                    % 为路口4设置不同的参数
                    params.AssignmentThreshold   = [12.0330472887580 35.4772047804730];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.850992633252816;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 5.12654967742099e-08;
                    params.NewTargetDensity      = 1.16335974991717e-07;
                    params.ConfirmationThreshold = 0.877240524328496;
                    params.DeletionThreshold     = 0.513808561871926;
                    params.DeathRate             = 0.447659502700140;
            
                case 'test_data_junc5'
                    % 为路口5设置不同的参数
                    params.AssignmentThreshold   = [9.64160061922475 22.9439881814141];
                    params.MaxNumTracks          = 500;
                    params.DetectionProbability  = 0.821602914776620;
                    params.MaxNumEvents          = 50;
                    params.ClutterDensity        = 1.26150090915959e-07;
                    params.NewTargetDensity      = 4.72626735460624e-07;
                    params.ConfirmationThreshold = 0.884787580569779;
                    params.DeletionThreshold     = 0.500236998917005;
                    params.DeathRate             = 0.633968708808360;
            
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