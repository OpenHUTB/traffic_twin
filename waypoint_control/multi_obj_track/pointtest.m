%data = load("690.mat");
%detector = load("trainedCustomPointPillarsDetector.mat");
currentPath = fileparts(mfilename('fullpath'));
detectModel = fullfile(currentPath,'trainedCustomPointPillarsDetector.mat');
pretrainedDetector = load(detectModel,'detector');
detector = pretrainedDetector.detector;

%ptCloud = datalog.LidarData.PointCloud;

ptCloud = pcread('2800.pcd');
classNames = {'car','truck','pedestrian'};
colors = {'green','magenta','blue'};

[bboxes,score,labels] = detect(detector,ptCloud);

helperShowPointCloudWith3DBoxes(ptCloud,bboxes,labels,classNames,colors)




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