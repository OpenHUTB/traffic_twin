pretrainedDetector = load('pretrainedPointPillarsDetector.mat','detector');
detector = pretrainedDetector.detector;

% 获取当前脚本所在的路径
currentFolder = fileparts(mfilename('fullpath'));

% 设置数据路径为当前脚本所在目录下的相对路径
dataPath = fullfile(currentFolder, 'mat格式', 'data');

% 获取目录下的所有 .mat 文件
matFiles = dir(fullfile(dataPath, "*.mat"));

% 遍历每个 .mat 文件
for fileIdx = 1:length(matFiles)
    % 加载当前 .mat 文件
    fileName = fullfile(dataPath, matFiles(fileIdx).name);
    load(fileName);
    fileName = fullfile(dataPath, matFiles(fileIdx).name);
    disp(fileName)
    % 检查变量是否存在并提取 datalog
    if exist('datalog', 'var') && isfield(datalog, 'LidarData') && isfield(datalog.LidarData, 'PointsCloud')
        points = datalog.LidarData.PointsCloud.Location;
        intensity = datalog.LidarData.PointsCloud.Intensity;
        % 检查点云数据是否为空
        if isempty(points) || isempty(intensity)
            warning('文件 %s 中的点云数据为空，跳过此文件。', fileName);
            continue;
        end

        % 创建点云对象并附加 Intensity 信息
        ptCloud = pointCloud(points, 'Intensity', intensity);
        % 增加检测的置信度阈值
        [bboxes, scores, labels] = detect(detector, ptCloud,  'Threshold', 0.25);
        datalog.LidarData.Detections = bboxes;
        % 保存更新后的数据到文件
        save(fileName, 'datalog', '-v7.3');
    else
        warning('文件 %s 中不存在有效的点云数据，跳过此文件。\n', fileName);
    end
end
