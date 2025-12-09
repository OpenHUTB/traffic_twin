% 将当前视角的相机2D框和融合框作匹配，保存2D框，轨迹ID,图片
function trkIDimg2DBox = extractTrackVehiclePicture(cameraBoxData, trkBox, trkIDs, categoryData)
    numBoxes = size(trkBox, 3); % 获取数组的尺寸
    trkIDimg2DBox = {};
    for k = 1:numBoxes
        currentBox = trkBox(:, :, k); % 获取当前融合框（8x2的子矩阵）
        % 计算x坐标的最小值
        minX = min(currentBox(:, 1));
        % 计算x坐标的最大值
        maxX = max(currentBox(:, 1));
        % 计算y坐标的最大值
        maxY = max(currentBox(:, 2));
        pos1 = [(minX+maxX)/2,maxY];
        % 初始化变量来保存最小距离和对应的检测框
        minDistance = inf; % 初始化为无穷大
        closestBox = [];
        category = [];
        numBoxes = size(cameraBoxData, 1);
        % 遍历yolo检测框
        for i = 1:numBoxes
            currentRow = cameraBoxData(i, :);
            pos2 = [currentRow(1)+currentRow(3)/2,currentRow(2)+currentRow(4)/2];
            % 计算两点之间的距离
            dx = pos2(1) - pos1(1);
            dy = pos2(2) - pos1(2);
            distance = sqrt(dx^2 + dy^2);
            % 更新最小距离和对应的检测框（如果当前距离更小）
            if distance < minDistance
                minDistance = distance;
                closestBox = currentRow;
                category = categoryData(i, :);
            end
        end
        
        % 将轨迹ID和对应的最匹配检测框保存
        trkIDimg2DBox{k} = struct('trkID', trkIDs(k), 'Box', closestBox, 'Category', category);
    end
end
