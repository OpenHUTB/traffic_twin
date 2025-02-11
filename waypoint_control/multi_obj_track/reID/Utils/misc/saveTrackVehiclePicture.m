function saveTrackVehiclePicture(trkIDimg2DBox, savedImg)
     currentPath = fileparts(mfilename('fullpath'));
     parentPath = fileparts(currentPath);
     grandparentPath = fileparts(parentPath);
     dataPath = fullfile(grandparentPath, 'trkIDImg');
     if ~exist(dataPath, 'dir')
        mkdir(dataPath);
     end
     
    for k = 1:numel(trkIDimg2DBox) % 遍历 trkIDimg2DBox 中的每个元素
        trkID = trkIDimg2DBox{k}.trkID; % 获取当前元素的轨迹ID
        closestBox = trkIDimg2DBox{k}.Box; % 获取当前元素的检测框（Box）
        outputImgPath = fullfile(dataPath, sprintf('%d.jpeg', trkID));  % 使用图片索引命名
        % 检查该文件是否已经存在
        if exist(outputImgPath, 'file')
            continue;  % 如果文件存在，跳过当前迭代
        end
        % 使用closestBox裁剪车辆区域
        croppedImg = imcrop(savedImg, closestBox);  % closestBox为[xmin, ymin, width, height]
        % 缩放到指定大小 224x224
        resizedImg = imresize(croppedImg, [224, 224]);
        % 保存裁剪后的图像
        
        imwrite(resizedImg, outputImgPath);
    end


end