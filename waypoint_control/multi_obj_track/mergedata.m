% 设置路径
folder1 = "processed_data/vhicle";
folder2 = "processed_data/pedestrian";
targetFolder = "total";

% 创建目标文件夹
if ~exist(targetFolder, 'dir')
    mkdir(targetFolder);
end

% 读取 folder1 中的编号文件夹
sub1 = dir(folder1);
sub1 = sub1([sub1.isdir] & ~ismember({sub1.name}, {'.','..'}));
sub1Names = natsortfiles({sub1.name});  % 自然排序
num1 = length(sub1Names);

% 将 folder1 中的子文件夹拷贝到目标中（保持原编号）
for i = 1:num1
    srcPath = fullfile(folder1, sub1Names{i});
    dstPath = fullfile(targetFolder, sub1Names{i});
    copyfile(srcPath, dstPath);
end

% 读取 folder2 中的编号文件夹
sub2 = dir(folder2);
sub2 = sub2([sub2.isdir] & ~ismember({sub2.name}, {'.','..'}));
sub2Names = natsortfiles({sub2.name});  % 自然排序

% 将 folder2 中的子文件夹重命名后拷贝过去
for j = 1:length(sub2Names)
    srcPath = fullfile(folder2, sub2Names{j});
    newIndex = num1 + j;
    dstPath = fullfile(targetFolder, num2str(newIndex));
    copyfile(srcPath, dstPath);
end

disp('文件夹合并完成');


function sorted = natsortfiles(names)
    nums = zeros(1, numel(names));
    for k = 1:numel(names)
        nums(k) = str2double(names{k});
    end
    [~, idx] = sort(nums);
    sorted = names(idx);
end