%% 使用训练的网络做车辆再识别
currentPath = fileparts(mfilename('fullpath'));
datasetFolder = "trainedCustomReidNetwork.mat";
netFolder = fullfile(currentPath ,datasetFolder);
% 加载训练的ReID网络,Re-ID此网络被训练来提取车辆的外观特征，并可以用来对同一车辆进行跨帧匹配
data = load(netFolder);
net = data.net;

% 加载预训练的对象检测器
name = "tiny-yolov4-coco";
detector = yolov4ObjectDetector(name);

% 再识别的图片文件夹路径
imageFolder = fullfile(currentPath ,"reidFile");  % 设置图片文件夹路径
imageFiles = dir(fullfile(imageFolder, "*.jpg"));  % 获取所有JPEG格式的图片文件，修改为适合的扩展名

vehicleMontage = {};  % 存储检测到的车辆图像的单元格数组，用于可视化检测到的车辆。
firstVehicleFature = [];  % 存储已知车辆的特征向量（Re-ID
for i = 1:numel(imageFiles)  % 遍历所有图像文件
    % 读当前帧
     % 读取当前图像
    imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
    vidFrame = imread(imgPath);

    % 使用车辆检测器（detector）对当前帧进行检测，返回检测到的边界框 bboxes、得分 scores 和标签 labels。
    [bboxes, scores, labels] = detect(detector,vidFrame,Threshold=0.5);
    % 用于裁剪边界框，确保其检测到的边界框不会超出图像的边界，在图像内
    bboxes = bboxcrop(bboxes,[1 1 size(vidFrame,2) size(vidFrame,1)]);
    % 将边界框坐标四舍五入为整数值。
    bboxes = round(bboxes);

    % 计算每个标签出现的次数
    numLabels = countcats(labels);
    % 假设车辆被标记为 "car"，numVehicle 存储当前帧中检测到的车辆数。
    numVehicle = numLabels(3); 

    % 为存储每个车辆提取的外观特征（Re-ID特征向量）创建一个二维数组（矩阵）
    appearanceData = zeros(2048,numVehicle);
    % 为存储每个检测到的车辆图像预分配一个单元格数组。
    croppedVehicle = cell(numVehicle);
    % 初始化一个计数器 vehicle，用于标记当前正在处理的车辆的索引
    vehicle = 1;
    % 遍历所有检测到的物体，若标签为 "car"，则裁剪出车辆的图像，并通过预训练的行人重识别网络（pretrainedNet）提取特征。
    for j = 1:size(bboxes,1)
        if labels(j) == "car" || labels(j) == "truck" % 若该物体的标签为 "car"或"truck"（即它是一个车辆），则进行裁剪并提取车辆图像的特征。
            % 获取当前车辆的边界框 bbox
            bbox = bboxes(j,:);
            % 根据 bbox 提供的边界框裁剪出当前车辆的图像
            croppedImg = imcrop(vidFrame,bbox);
            % 对裁剪出来的图像 croppedImg 进行缩放，将其调整为 224x224 的大小
            % 并将缩放后的图像存储到 croppedVehicle 单元格数组的第 vehicle 个元素中
            croppedVehicle{vehicle} = imresize(croppedImg,[224 224]);
            % 提取该裁剪图像的外观特征
            appearanceData(:,vehicle) = extractReidentificationFeatures(net,croppedImg);
            vehicle = vehicle + 1;
        end
    end

    if i == 1
        firstVehicleFature = appearanceData(:,1);
        vehicleMontage{end+1} = croppedVehicle{1};
    else
        
        % 如果 vehicleFeature 已经存储了某车辆的特征，则使用余弦相似度（pdist2）
        % 计算当前帧中每个车辆特征与已知车辆特征的相似度。
        cosineSimilarity = 1-pdist2(firstVehicleFature',appearanceData',"cosine");
        % 选择相似度最大（即匹配度最好的车辆）的索引 matchIdx 和相似度值 cosSim。
        [cosSim,matchIdx] = max(cosineSimilarity);
               
        % 设定一个相似度阈值 similarityThreshold，若当前帧中的最匹配车辆的相似度高于该阈值，则认为该车辆是同一辆车
        % 更新 vehicleFeature 为当前帧中最匹配车辆的特征，并将该车辆的图像添加到 vehicleMontage 中。
        similarityThreshold = 0.5;
        if cosSim > similarityThreshold
           disp(cosSim)
           firstVehicleFature = appearanceData(:,matchIdx);
           vehicleMontage{end+1} = croppedVehicle{matchIdx};
        end
         
    end

end
% 显示整个图像序列中识别的车辆。
montage(vehicleMontage)


