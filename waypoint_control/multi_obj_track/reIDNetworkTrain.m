%% 重新训练ReID网络
% 加载数据并准备训练集与验证集
currentPath = fileparts(mfilename('fullpath'));
datasetFolder = "processed_data";
dataFolder = fullfile(currentPath ,datasetFolder);
imds = imageDatastore(dataFolder,IncludeSubfolders=true,LabelSource="foldernames");

% 设置随机数生成器的种子为 0
rng(0)
%  将数据集中的图像顺序打乱。
ds = shuffle(imds);

% 将 90% 的数据用作训练集
numTraining = round(size(imds.Files,1)*0.9);
% 获取前 90% 的数据作为训练集。
dsOcclude = subset(ds,1:numTraining);
% 获取剩余 10% 的数据作为验证集。
dsVal = subset(ds,numTraining+1:size(imds.Files,1));

%% 生成遮挡数据
% 指定用于存储遮挡数据的文件夹。
occlusionDatasetDirectory = fullfile(currentPath, "vehicleOcclusionDataset");
% 检查 generateOcclusionData 是否存在，如果不存在则设置其值为 true，表示需要生成遮挡数据
if ~exist("generateOcclusionData","var")
    generateOcclusionData = true;
end
% 如果 generateOcclusionData 为 true 且遮挡数据目录不存在，则调用 writeall
% 函数将训练集数据保存到指定文件夹，并且使用 helperGenerateOcclusionData 函数为数据添加遮挡。
if generateOcclusionData && ~exist(occlusionDatasetDirectory,"dir")
    writeall(dsOcclude,occlusionDatasetDirectory,WriteFcn=@(img,writeInfo,outputFormat) ...
        helperGenerateOcclusionData(img,writeInfo,outputFormat,datasetFolder));
    generateOcclusionData = false;
end

%% 训练数据预览
dsTrain = imageDatastore(fullfile(occlusionDatasetDirectory,datasetFolder),IncludeSubfolders=true,LabelSource="foldernames");

previewImages = cell(1,4);
for i = 1:4
    previewImages{i} = readimage(dsTrain,randi(numel(dsTrain.Files)));
end
montage(previewImages,Size=[1 4])

%% 训练网络
% 重新初始化训练数据集，确保每次从头开始。
reset(dsTrain)
% 使用预训练的 ResNet-50 网络作为骨干网络。
resbackbone = imagePretrainedNetwork("resnet50");
% 获取所有不同的类标签
classes = unique(imds.Labels);
%  定义输入图像的大小
inputSize = [224 224 3];
% 定义特征向量的长度。车辆：4096
featureLength = 2048;
% 使用 reidentificationNetwork 函数创建一个行人重识别网络，使用 ResNet-50 作为骨干网络。
net = reidentificationNetwork(resbackbone,classes,InputSize=inputSize,FeatureLength=featureLength);
numEpochs = 120;
miniBatchSize = 64;

% 定义训练选项并训练模型
options = trainingOptions("sgdm", ...
    MaxEpochs=numEpochs, ...
    ValidationData=dsVal, ...
    InitialLearnRate=0.01, ...
    LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=round(numEpochs/2), ...
    LearnRateSchedule="piecewise", ...
    MiniBatchSize=miniBatchSize, ...
    OutputNetwork="best-validation", ...
    Shuffle="every-epoch", ...
    VerboseFrequency=30, ...
    Verbose=true);
% 开始训练网络，使用余弦-Softmax 损失函数，并且不冻结骨干网络。
net = trainReidentificationNetwork(dsTrain,net,options, ...
        LossFunction="cosine-softmax",FreezeBackbone=false);


% 保存训练好的模型
modelFilePath = fullfile(currentPath, 'trainedCustomReidNetwork.mat');
save(modelFilePath, 'net');


function helperGenerateOcclusionData(img,writeInfo,~,datasetFolder)

info = writeInfo.ReadInfo;
occlusionDataFolder = writeInfo.Location;

% Get the name of the training image.
fileName = info.Filename;

% Find the last slash in the image filename path to extract
% only the actual image file name.
if ispc
    slash = "\";
else
    slash = "/";
end
slashIdx = strfind(info.Filename,slash);
imgName = info.Filename(slashIdx(end)+1:end);

% Set the output folder for the given indentity.
imagesDataFolder = fullfile(occlusionDataFolder,datasetFolder,string(info.Label));

% Copy the original file to the occlusion training data folder if it
% does not already exist.
if ~isfile(fullfile(imagesDataFolder,imgName))
    copyfile(fileName,fullfile(imagesDataFolder,imgName));
end

numBlockErases = 3;

for i = 1:numBlockErases
    % Select a random square window from the image. The area of the window is
    % between 4% and 15% of the area of the entire image.
    win = randomWindow2d(size(img),Scale=[0.04 0.15],DimensionRatio=[1 3;1 1]);

    % Determine the height and width of the erase region.
    hwin = diff(win.YLimits)+1;
    wwin = diff(win.XLimits)+1;

    % Erase the pixels within the erase region. Fill each pixel with a random color.
    img = imerase(img,win,FillValues=randi([1 255],[hwin wwin 3]));
end

imwrite(img,fullfile(imagesDataFolder,strcat(imgName(1:end-5),"_occlusion.jpeg")));

end
