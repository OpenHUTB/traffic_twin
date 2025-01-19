% 数据路径
currentPath = fileparts(mfilename('fullpath'));
detectModel = fullfile(currentPath,'trainedCustomPointPillarsDetector.mat');
pretrainedDetector = load(detectModel,'detector');
detector = pretrainedDetector.detector;

dataPath = fullfile(currentPath, 'test_data');

% 获取目录下的所有 .mat 文件
matFiles = dir(fullfile(dataPath, "*.mat"));

% 遍历每个 .mat 文件
for fileIdx = 1:length(matFiles)
    % 加载当前 .mat 文件
    fileName = fullfile(dataPath, matFiles(fileIdx).name);
    load(fileName);
    fileName = fullfile(dataPath, matFiles(fileIdx).name);
    % 检查变量是否存在并提取 datalog
    if exist('datalog', 'var') && isfield(datalog, 'LidarData') && isfield(datalog.LidarData, 'PointCloud')
        points = datalog.LidarData.PointCloud.Location;
        intensity = datalog.LidarData.PointCloud.Intensity;
        % 检查点云数据是否为空
        if isempty(points) || isempty(intensity)
            warning('文件 %s 中的点云数据为空，跳过此文件。', fileName);
            continue;
        end

        % 创建点云对象并附加 Intensity 信息
        ptCloud = pointCloud(points, 'Intensity', intensity);
        % 将点云重组成有序的
        horizontalResolution = 1024;
        params = lidarParameters('HDL64E',horizontalResolution);
        ptCloudOrg = pcorganize(ptCloud,params);

        % 增加检测的置信度阈值(路口1使用的0.5)
        [bboxes, scores, labels] = detect(detector, ptCloudOrg,  'Threshold', 0.33);
        disp('Bounding Boxes:');
        disp(bboxes);
        
        disp('Scores:');
        disp(scores);
        
        disp('Labels:');
        disp(labels);
        datalog.LidarData.Detections = bboxes;
        % 保存更新后的数据到文件
        save(fileName, 'datalog', '-v7.3');
    else
        warning('文件 %s 中不存在有效的点云数据，跳过此文件。\n', fileName);
    end
end
