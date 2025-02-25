function detect2DBoundingBox(junc)
    % 数据路径
    currentPath = fileparts(mfilename('fullpath'));
    dataPath = fullfile(currentPath, junc);
    
    % 获取目录下的所有 .mat 文件
    matFiles = dir(fullfile(dataPath, "*.mat"));
    
    % 初始化 YOLOv4 检测器
    name = "tiny-yolov4-coco";
    detector = yolov4ObjectDetector(name);
    
    % 遍历每个 .mat 文件
    for fileIdx = 1:length(matFiles)
        % 加载当前 .mat 文件
        fileName = fullfile(dataPath, matFiles(fileIdx).name);
        load(fileName);
    
        % 检查变量是否存在并提取 datalog
        if exist('datalog', 'var')
            cameraData = datalog.CameraData;
        else
            warning('文件 %s 中不存在 datalog 变量，跳过此文件。', fileName);
            continue;
        end
    
        % 遍历 CameraData 中的每个相机数据
        for i = 1:numel(cameraData)
            % 获取图片路径
            imgPath = cameraData(i).ImagePath;
            fullImgPath = fullfile(dataPath, imgPath); % 构造完整路径
            disp(fullImgPath)
            
            if ~isfile(fullImgPath)
                warning('文件 %s 不存在，跳过此项。', fullImgPath);
                continue;
            end
    
            % 读取图片
            img = imread(fullImgPath);
            % 运行 YOLOv4 检测器
            [bboxes, scores, labels] = detect(detector, img);
             % 检测 labels 的类型并转换为字符数组（如果为 categorical 类型）
            if iscategorical(labels)
                labels = cellstr(labels); % 转换为字符数组
            end
            disp(labels)
         
            if isempty(labels)
                cameraData(i).Detections = zeros(0, 4); % 如果无检测结果，保存空矩阵
            else 
                % 过滤掉 labels 为 "traffic light" 的检测结果
                validIndices = ~strcmp(labels, 'traffic light');
                disp(validIndices)
                filteredBboxes = bboxes(validIndices, :);
        
                % 将检测结果保存到 Detections 字段
                if isempty(filteredBboxes)
                    cameraData(i).Detections = zeros(0, 4); % 如果无检测结果，保存空矩阵
                else
                    cameraData(i).Detections = filteredBboxes;
                end
            end
    
            % 可选：在图片上绘制检测框并显示
            % detectedImg = insertObjectAnnotation(img, "Rectangle", bboxes, labels);
            % figure;
            % imshow(detectedImg);
            % title(sprintf('File %s - Camera %d: Detected Objects', matFiles(fileIdx).name, i));
        end
    
        % 更新 datalog.CameraData
        datalog.CameraData = cameraData;
    
        % 保存结果回原始文件
        save(fileName, 'datalog', '-v7.3'); % '-v7.3' 确保兼容性
      
        % fprintf('文件 %s 的检测完成，结果已保存到 %s\n', matFiles(fileIdx).name, outputFileName);
    end
    
    fprintf('所有文件处理完成。\n');
end 