%% 裁剪在carla中收集的再识别数据集，使得能够在matlab中训练

currentPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(currentPath, 'reid_data/pedestrian/');
outputPath = fullfile(currentPath, 'processed_data/pedestrian');
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
% 获取所有的行人类型文件夹
pedestrianFolders = dir(dataPath);
pedestrianFolders = pedestrianFolders([pedestrianFolders.isdir] & ~ismember({pedestrianFolders.name}, {'.', '..'}));

% 遍历所有的类型文件夹
for i = 1:length(pedestrianFolders)
    pedestrianFolder = fullfile(dataPath, pedestrianFolders(i).name);
    
    % 获取与行人相关的camera1、camera2文件夹
    viewFolders = dir(pedestrianFolder);
    viewFolders = viewFolders([viewFolders.isdir] & ismember({viewFolders.name}, {'camera1', 'camera2'}));
    % 获取与行人相关的1.mat和2.mat文件
    matFiles = dir(fullfile(pedestrianFolder, '*.mat'));
    % 载入1.mat和2.mat文件
    matFile1 = load(fullfile(pedestrianFolder, matFiles(1).name));  % 1.mat
    matFile2 = load(fullfile(pedestrianFolder, matFiles(2).name));  % 2.mat
    
    pedestrianBboxes1 = matFile1.LabelData.Label;  % 载入camera1视角的标签数据
    pedestrianBboxes2 = matFile2.LabelData.Label;  % 载入camera2视角的标签数据
    % 遍历camera1和camera2视角文件夹
    for j = 1:length(viewFolders)
        viewFolder = fullfile(pedestrianFolder, viewFolders(j).name);
        
        % 获取该视角文件夹下的所有JPEG图片
        images = dir(fullfile(viewFolder, '*.jpeg'));  % 获取所有JPEG格式的图片文件
        
        % 确定该视角使用哪一个.mat文件的标签
        if strcmp(viewFolders(j).name, 'camera1')
            pedestrianBboxes = pedestrianBboxes1;  % camera1使用1.mat的标签
        else
            pedestrianBboxes = pedestrianBboxes2;  % camera2使用2.mat的标签
        end
        
        % 为每个行人类型和视角创建输出文件夹
        pedestrianOutputFolder = fullfile(outputPath, pedestrianFolders(i).name);
        if ~exist(pedestrianOutputFolder, 'dir')
            mkdir(pedestrianOutputFolder);
        end
        
        % 遍历每张图片
        for k = 1:length(images)
            imgPath = fullfile(viewFolder, images(k).name);
            img = imread(imgPath);
            
            % 获取当前图片的边界框
            bbox = pedestrianBboxes(k, :);  % 假设pedestrianBboxes是一个 [numImages x 4] 的数组
            
            % 使用bbox裁剪行人区域
            croppedImg = imcrop(img, bbox);  % bbox为[xmin, ymin, width, height]
            
            % 缩放到指定大小 224x224
            resizedImg = imresize(croppedImg, [224, 224]);
            
            % 保存裁剪后的图像
            outputImgPath = fullfile(pedestrianOutputFolder, sprintf('%d_%d.jpeg', j, k));  % 使用图片索引命名
            imwrite(resizedImg, outputImgPath);
        end
    end

end

disp('Data preprocessing complete!');