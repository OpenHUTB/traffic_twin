%% Step 1: 数据准备
% 设置自定义数据路径
% 数据路径
dataPath = fileparts(mfilename('fullpath'));
lidarDataPath = fullfile(dataPath, 'Lidar'); % 雷达点云文件夹
labelDataPath = fullfile(dataPath, 'Cuboids'); % 标签文件路径

% 加载点云数据(创建一个文件数据存储对象，用于从目录中逐个加载文件)
lidarData = fileDatastore(lidarDataPath, 'ReadFcn', @(x) pcread(x));

% 加载边界框标签
gtPath = fullfile(labelDataPath,'CarlaSetLidarGroundTruth.mat');
data = load(gtPath, 'LabelData'); 
Labels = data.LabelData;
%{
data = load(gtPath, 'LabelData'); 
timeTable = data.gTruth.LabelData;
% 转换成普通表格
Labels = timetable2table(timeTable);
%}
% 2、3、4列数据
boxLabels = Labels(:,2:4);

% 显示全视图点云
figure
ptCld = preview(lidarData);
ax = pcshow(ptCld.Location);
set(ax,'XLim',[-50 50],'YLim',[-40 40]);
zoom(ax,2.5);
axis off;

%% Step 2: 数据预处理
% 定义裁剪参数

xMin = -69.12;  % X 轴最小值
yMin = -39.68;  % Y 轴最小值
zMin = -5.0;    % Z 轴最小值
xMax = 69.12;   % X 轴最大值
yMax = 39.68;   % Y 轴最大值
zMax = 5.0;     % Z 轴最大值

pointCloudRange = [xMin xMax yMin yMax zMin zMax];

% 裁剪点云并处理标签
[croppedPointCloudObj, processedLabels] = cropFrontViewFromLidarData(...
    lidarData, boxLabels, pointCloudRange);

% 显示裁剪的点云和实际方码框标签。
pc = croppedPointCloudObj{1,1};
% processedLabels显示一帧的,也就是一行
% 判断 processedLabels.truck 的类型
if iscell(processedLabels.car)
    processcar = processedLabels.car{1};
else 
    processcar = processedLabels.car(1, :);
end
if iscell(processedLabels.truck)
    processtruck = processedLabels.truck{1};
else
    processtruck = processedLabels.truck(1, :);
end
if iscell(processedLabels.pedestrian)
    processpedestrian = processedLabels.pedestrian{1};
else
    processpedestrian = processedLabels.pedestrian(1, :);
end
bboxes = [processcar; processtruck; processpedestrian];
ax = pcshow(pc);
showShape('cuboid',bboxes,'Parent',ax,'Opacity',0.1,...
        'Color','green','LineWidth',0.5);
reset(lidarData);
%% Step 3: 创建数据存储对象
% 将数据集拆分为训练集和测试集。选择 80% 的数据用于训练网络，其余的数据用于评估。
% 设置随机种子
rng(1);
shuffledIndices = randperm(size(processedLabels,1)); % 生成一个随机的排列索引，对应到每个数据集。
idx = floor(0.8 * length(shuffledIndices));  % ：计算训练集的大小。

trainData = croppedPointCloudObj(shuffledIndices(1:idx),:);
testData = croppedPointCloudObj(shuffledIndices(idx+1:end),:);

trainLabels = processedLabels(shuffledIndices(1:idx),:);
testLabels = processedLabels(shuffledIndices(idx+1:end),:);
% 将第2列变成元胞数组
% 将 testLabels.truck 转换为元胞数组
% trainLabels.truck = mat2cell(trainLabels.truck, ones(size(trainLabels.truck, 1), 1), size(trainLabels.truck, 2));
% testLabels.truck = mat2cell(testLabels.truck, ones(size(testLabels.truck, 1), 1), size(testLabels.truck, 2));

% 训练数据（点云及其对应的标签）保存为 PCD 文件，以便之后能够方便地加载和访问这些文件。
writeFiles = true;
% 确保目标文件夹存在
dataLocation = fullfile(dataPath, 'InputData');
if ~exist(dataLocation, 'dir')
    mkdir(dataLocation); % 创建文件夹
end
% 将 trainData 和 trainLabels 保存为 PCD 文件，并返回新的训练数据和标签
[trainData,trainLabels] = saveptCldToPCD(trainData,trainLabels,...
    dataLocation,writeFiles);
                 
% 创建文件数据存储，加载 PCD 文件（点云数据）。
lds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));
% 创建框标签数据存储，以加载 3-D 边界框标签。
bds = boxLabelDatastore(trainLabels);
% 数将点云和 3-D 边界框标签合并到单个数据存储中进行训练。
cds = combine(lds,bds);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
%% 执行数据增强，通过过采样和变换对点云数据进行增强，增加数据集的多样性
% 在增强之前使用示例末尾定义的辅助函数读取并显示点云
augData = preview(cds); %  预览 cds 数据存储对象(一行)
classNames = {'car','truck','pedestrian'};
colors = {'green','magenta','blue'};
[ptCld,bboxes,labels] = deal(augData{1},augData{2},augData{3});
% 显示预览点云视图
helperShowPointCloudWith3DBoxes(ptCld,bboxes,labels,classNames,colors)

% 定义了一个文件夹路径,用于存储采样后的点云数据和标签
sampleLocation = fullfile(dataPath,'GTsamples');
if ~exist(sampleLocation, 'dir')
    mkdir(sampleLocation); % 创建文件夹
end
% 从给定的数据存储对象 cds 中进行数据采样;返回两个输出,包含采样后的点云数据的存储对象和边界框标签的存储对象。
[ldsSampled,bdsSampled] = sampleLidarData(cds,classNames,'MinPoints',15,...                  
                            'Verbose',false,'WriteLocation',sampleLocation);
cdsSampled = combine(ldsSampled,bdsSampled); % 合并采样数据
%  数据增强：过采样
numObjects = [20 20 20]; % 定义了每种类别（如车和卡车）要进行的过采样次数。
% 使用 transform 函数对 cds 数据进行数据增强，增加每个类别的样本数量，改善数据的平衡性。
cdsAugmented = transform(cds,@(x)pcBboxOversample(x,cdsSampled,classNames,numObjects));
% 数据增强：变换
% 对已增强的数据 cdsAugmented 进行进一步的处理
% 进一步增加数据多样性，防止过拟合，并提高模型的鲁棒性。
cdsAugmented = transform(cdsAugmented,@(x)helperAugmentData(x));

augData = preview(cdsAugmented); % 预览增强后的数据
[ptCld,bboxes,labels] = deal(augData{1},augData{2},augData{3});
helperShowPointCloudWith3DBoxes(ptCld,bboxes,labels,classNames,colors)

%% 创建 PointPillars 对象检测器
anchorBoxes = calculateAnchorsPointPillars(trainLabels);
voxelSize = [0.16, 0.16]; % 体素大小

% 创建 PointPillars 模型
detector = pointPillarsObjectDetector(pointCloudRange, classNames, ...
    anchorBoxes, 'VoxelSize', voxelSize);

%% Step 4: 设置训练选项并训练模型
checkpointPath = fullfile(dataPath,'checkpointPath');
if ~exist(checkpointPath, 'dir')
    mkdir(checkpointPath); % 创建文件夹
end
executionEnvironment = "auto"; % 或者设置为 "gpu" 或 "cpu"
options = trainingOptions('adam',...        % 指定使用的优化器为 Adam（自适应矩估计
    Plots = "training-progress",...         % 参数表示训练过程中绘制图表以显示训练进度
    MaxEpochs = 60,...                      % 指定训练的最大周期数（epochs）
    MiniBatchSize = 3,...                   % 每次训练时使用的小批量（mini-batch）大小
    GradientDecayFactor = 0.9,...           % Adam 优化算法中的一个参数，控制梯度的衰减因子。
    SquaredGradientDecayFactor = 0.999,...  % Adam 优化器中的一个参数，控制平方梯度的衰减因子。
    LearnRateSchedule = "piecewise",...     % 设置学习率调度方式。"piecewise" 意味着学习率会按预设的计划在某些周期后下降。
    InitialLearnRate = 0.0002,...           % 初始学习率用于控制每次更新模型参数时的步长
    LearnRateDropPeriod = 15,...            % 该参数表示学习率下降的周期数。
    LearnRateDropFactor = 0.8,...           % 学习率下降因子。
    ExecutionEnvironment= executionEnvironment, ... 
    PreprocessingEnvironment = 'parallel',...      % 指定是否启用数据并行预处理。如果设置为 'parallel'，则在数据加载时启用并行处理，以加速数据预处理。
    BatchNormalizationStatistics = 'moving',...    % 指定批量归一化（Batch Normalization）使用的统计量类型
    ResetInputNormalization = false,...            % 如果设置为 true，则每次训练时会重置输入数据的归一化参数
    CheckpointFrequency = 10, ...                  % 该参数设置每隔多少周期保存一次模型检查点
    CheckpointFrequencyUnit = 'epoch', ...         % 这个参数指定检查点保存频率的单位
    CheckpointPath = checkpointPath);              % 数指定保存模型检查点的路径。

% 训练模型
[detector, info] = trainPointPillarsObjectDetector(cdsAugmented, ...
        detector, options);

%% Step 5: 测试模型并可视化结果
% 从测试集中取出一个点云
ptCloud = testData{1,1};
[bboxes,score,labels] = detect(detector,ptCloud);
helperShowPointCloudWith3DBoxes(ptCloud,bboxes,labels,classNames,colors)

%% Step 6: 保存模型
dataPath = fileparts(mfilename('fullpath'));  % 获取当前脚本所在的文件夹路径
save(fullfile(dataPath, 'trainedCustomPointPillarsDetector.mat'), 'detector');  % 保存文件到 dataPath 目录
%% 辅助函数
function data = helperAugmentData(data)
    % Apply random scaling, rotation and translation.
    pc = data{1};
    
    minAngle = -45;
    maxAngle = 45;
    
    % Define outputView based on the grid-size and XYZ limits.
    outView = imref3d([32,32,32],[-100,100],...
        [-100,100],[-100,100]);
    
    
    theta = minAngle + rand(1,1)*(maxAngle - minAngle);
    tform = randomAffine3d('Rotation',@() deal([0,0,1],theta),...
        'Scale',[0.95,1.05],...
        'XTranslation',[0,0.2],...
        'YTranslation',[0,0.2],...
        'ZTranslation',[0,0.1]);
    
    % Apply the transformation to the point cloud.
    ptCloudTransformed = pctransform(pc,tform);
    
    % Apply the same transformation to the boxes.
    bbox = data{2};
    [bbox,indices] = bboxwarp(bbox,tform,outView);
    if ~isempty(indices)
        data{1} = ptCloudTransformed;
        data{2} = bbox;
        data{3} = data{1,3}(indices,:);
    end

end

function helperShowPointCloudWith3DBoxes(ptCld,bboxes,labels,classNames,colors)
    % Validate the length of classNames and colors are the same
    assert(numel(classNames) == numel(colors), 'ClassNames and Colors must have the same number of elements.');
    
    % Get unique categories from labels
    uniqueCategories = categories(labels); 
    disp(uniqueCategories)
    % Create a mapping from category to color
    colorMap = containers.Map(uniqueCategories, colors); 
    labelColor = cell(size(labels));

    % Populate labelColor based on the mapping
    for i = 1:length(labels)
        labelColor{i} = colorMap(char(labels(i)));
    end

    figure;
    ax = pcshow(ptCld); 
    showShape('cuboid', bboxes, 'Parent', ax, 'Opacity', 0.1, ...
        'Color', labelColor, 'LineWidth', 0.5);
    zoom(ax,1.5);
end