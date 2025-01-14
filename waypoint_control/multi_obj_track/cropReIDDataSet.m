%% 裁剪在carla中收集的再识别数据集，使得能够在matlab中训练

currentPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(currentPath, 'reid_data');
outputPath = fullfile(currentPath, 'processed_data');
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
% 获取所有的车辆类型文件夹
vehicleFolders = dir(dataPath);
vehicleFolders = vehicleFolders([vehicleFolders.isdir] & ~ismember({vehicleFolders.name}, {'.', '..'}));

% 遍历所有的车辆类型文件夹
for i = 1:length(vehicleFolders)
    vehicleFolder = fullfile(dataPath, vehicleFolders(i).name);
    
    % 获取与车辆相关的camera1、camera2文件夹
    viewFolders = dir(vehicleFolder);
    viewFolders = viewFolders([viewFolders.isdir] & ismember({viewFolders.name}, {'camera1', 'camera2'}));
    % 获取与车辆相关的1.mat和2.mat文件
    matFiles = dir(fullfile(vehicleFolder, '*.mat'));
    % 载入1.mat和2.mat文件
    matFile1 = load(fullfile(vehicleFolder, matFiles(1).name));  % 1.mat
    matFile2 = load(fullfile(vehicleFolder, matFiles(2).name));  % 2.mat
    vehicleBboxes1 = matFile1.LabelData.Label;  % 载入camera1视角的标签数据
    vehicleBboxes2 = matFile2.LabelData.Label;  % 载入camera2视角的标签数据
    % 遍历camera1和camera2视角文件夹
    for j = 1:length(viewFolders)
        viewFolder = fullfile(vehicleFolder, viewFolders(j).name);
        
        % 获取该视角文件夹下的所有JPEG图片
        images = dir(fullfile(viewFolder, '*.jpeg'));  % 获取所有JPEG格式的图片文件
        
        % 确定该视角使用哪一个.mat文件的标签
        if strcmp(viewFolders(j).name, 'camera1')
            vehicleBboxes = vehicleBboxes1;  % camera1使用1.mat的标签
        else
            vehicleBboxes = vehicleBboxes2;  % camera2使用2.mat的标签
        end
        
        % 为每个车辆类型和视角创建输出文件夹
        vehicleOutputFolder = fullfile(outputPath, vehicleFolders(i).name);
        if ~exist(vehicleOutputFolder, 'dir')
            mkdir(vehicleOutputFolder);
        end
        
        % 遍历每张图片
        for k = 1:length(images)
            imgPath = fullfile(viewFolder, images(k).name);
            img = imread(imgPath);
            
            % 获取当前图片的边界框
            bbox = vehicleBboxes(k, :);  % 假设vehicleBboxes是一个 [numImages x 4] 的数组
            
            % 使用bbox裁剪车辆区域
            croppedImg = imcrop(img, bbox);  % bbox为[xmin, ymin, width, height]
            
            % 缩放到指定大小 224x224
            resizedImg = imresize(croppedImg, [224, 224]);
            
            % 保存裁剪后的图像
            outputImgPath = fullfile(vehicleOutputFolder, sprintf('%d_%d.jpeg', j, k));  % 使用图片索引命名
            imwrite(resizedImg, outputImgPath);
        end
    end

end

disp('Data preprocessing complete!');