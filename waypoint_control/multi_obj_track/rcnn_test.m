%% 下载数据集
zipFile = matlab.internal.examples.downloadSupportFile("lidar","data/PandasetLidarData.zip");
saveFolder = fileparts(zipFile);
unzip(zipFile,saveFolder)
outputFolder = fullfile(saveFolder,"PandasetLidarData");

%% 加载数据
path = fullfile(outputFolder,"PointCloud");
lds = fileDatastore(path,"ReadFcn",@(x) pcread(x));

gtPath = fullfile(outputFolder,"Cuboid","PandaSetLidarGroundTruth.mat");
bboxLabels = load(gtPath,"lidarGtLabels");
bboxTable = bboxLabels.lidarGtLabels;

pointCloudRange = [0 70.4 -40 40 -3 1];

bboxTable = helperProcessGroundTruthData(bboxTable,pointCloudRange);

bds = boxLabelDatastore(bboxTable);

cds = combine(lds,bds);

rng(8)
totalElements = numpartitions(cds);

% 生成一个随机排列的索引
shuffledIndices = randperm(totalElements);

% 指定拆分比例，并计算训练元素的数量
trainRatio = 0.8;
numTrainElements = round(totalElements*trainRatio);

% 确定训练集和测试集的索引
trainIndices = shuffledIndices(1:numTrainElements);
testIndices = shuffledIndices(numTrainElements+1:end);

% 根据计算出的索引创建数据存储的子集
trainDs = subset(cds,trainIndices);
testDs = subset(cds,testIndices);

trainingSample = preview(trainDs);
[ptCld,bboxes,labels] = deal(trainingSample{1},trainingSample{2},trainingSample{3});

% 定义用于目标检测的类
classNames = {'Car','Truck','Pedestrian'};

% 为每个类别定义颜色以绘制边界框
colors = {'green','magenta','yellow'};

helperShowPointCloudWith3DBoxes(ptCld,bboxes,labels,classNames,colors)

%% 数据增强
sampleLocation = fullfile(outputFolder,"GTsamples");
writeFiles = true;
if writeFiles
    [ldsSampled,bdsSampled] = sampleLidarData(trainDs,classNames,MinPoints=[20 20 10], ...                  
                                Verbose=false,WriteLocation=sampleLocation);
    cdsSampled = combine(ldsSampled,bdsSampled);
    save(fullfile(sampleLocation,"augmentedSample"),"cdsSampled")
else
    load(fullfile(sampleLocation,"augmentedSample"))
end

numObjects = 10;
cdsAugmented = transform(trainDs,@(x)pcBboxOversample(x,cdsSampled,classNames,numObjects));

cdsAugmented = transform(cdsAugmented,@(x)helperAugmentData(x));

augData = preview(cdsAugmented);
[ptCld,bboxes,labels] = deal(augData{1},augData{2},augData{3});
helperShowPointCloudWith3DBoxes(ptCld,bboxes,labels,classNames,colors)

%% 创建 Voxel R-CNN 对象检测器
anchorBoxes = helperEstimate3DAnchorBoxesForVoxelRCNN(trainDs,classNames);

detector = voxelRCNNObjectDetector("kitti",classNames,anchorBoxes,PointCloudRange=pointCloudRange); 

%% 设置训练选项并训练模型

options = trainingOptions("adam", ...
    InitialLearnRate=0.01, ...
    MiniBatchSize=4, ...
    MaxEpochs=10, ...
    PreprocessingEnvironment="parallel", ...
    VerboseFrequency=100, ...
    CheckpointFrequency=10, ...
    CheckpointPath=userpath, ...
    Plots="training-progress", ...
    Shuffle="every-epoch", ...
    L2Regularization=0.01);

[detector,info] = trainVoxelRCNNObjectDetector(cdsAugmented,detector,options);

%% 测试模型并可视化结果
testData = preview(testDs);
ptCloud = testData{1};
[bboxes,score,labels] = detect(detector,ptCloud);

helperShowPointCloudWith3DBoxes(ptCloud,bboxes,labels,classNames,colors)

%%  保存模型
dataPath = fileparts(mfilename('fullpath'));  % 获取当前脚本所在的文件夹路径
save(fullfile(dataPath, 'trainedCustomVoxelRCNNDetector.mat'), 'detector');  % 保存文件到 dataPath 目录
%% 辅助函数
function helperShowPointCloudWith3DBoxes(ptCld,bboxes,labels,classNames,colors)
    % Validate the length of classNames and colors are the same.
    assert(numel(classNames)==numel(colors),"ClassNames and Colors must have the same number of elements.")
    
    % Get unique categories from labels.
    uniqueCategories = categories(labels); 

    % Create a mapping from category to color.
    colorMap = containers.Map(uniqueCategories,colors); 
    labelColor = cell(size(labels));

    % Populate labelColor based on the mapping.
    for i = 1:length(labels)
        labelColor{i} = colorMap(char(labels(i)));
    end

    figure
    ax = pcshow(ptCld); 
    showShape("cuboid",bboxes,Parent=ax,Opacity=0.1, ...
        Color=labelColor,LineWidth=0.5)
    zoom(ax,5)
  
end


function data = helperAugmentData(data)
    pc = data{1};
    
    % Define outputView based on the grid-size and XYZ limits.
    outView = imref3d([32,32,32],[-100,100], ...
        [-100,100],[-100,100]);
    rotationAngle = [-45, 45];
    rotationNoise = (rotationAngle(2) - rotationAngle(1)).*rand() + rotationAngle(1);
    tform = randomAffine3d(Rotation=@() deal([0,0,1],rotationNoise), ...
        Scale=[0.95 1.05], ...
        XTranslation=[0 0.2], ...
        YTranslation=[0 0.2], ...
        ZTranslation=[0 0.1]);
    
    ptCloud = pctransform(pc,tform);
    % Apply the same transformation to the boxes.
    bbox = data{2};
    [bbox,indices] = bboxwarp(bbox,tform,outView);
    if ~isempty(indices)
        data{1} = ptCloud;
        data{2} = bbox;
        data{3} = data{1,3}(indices,:);
    end
end


function boxTable = helperProcessGroundTruthData(boxTable,pcRange)
for colIdx = 1:size(boxTable,2)
    % Apply the operation to each cell in the current column.
    boxTable{:,colIdx} = cellfun(@(x) helperGetBbox(x,pcRange),boxTable{:,colIdx},UniformOutput=false);
end
end


function data = helperGetBbox(data,pcRange)
    bbox = data;
    if ~isempty(bbox)
        % Get index of bounding boxes that lie within pcRange.
        idx = (bbox(:,1) > pcRange(1)) & ...
            (bbox(:,1) < pcRange(2)) & ...
            (bbox(:,2) > pcRange(3)) & ...
            (bbox(:,2) < pcRange(4));
        data = bbox(idx,:);
    end
end


function data = helperGetRotatedRectangle(data)
bbox = data{1};
if ~isempty(bbox)
    % Convert bounding box format to rotated rectangle.
    data{1} = bbox(:,[1 2 4 5 9]);
end
end