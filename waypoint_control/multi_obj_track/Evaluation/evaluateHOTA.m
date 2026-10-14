function hotaTable = evaluateHOTA(tracks, truths, varargin)
%evaluateHOTA 计算 HOTA 多目标跟踪指标（HOTA, DetA, AssA, LocA）
%   参考：Luiten et al., "HOTA: A Higher Order Metric for Evaluating
%   Multi-object Tracking", IJCV 2021。
%
%   输入：
%     tracks - 结构体数组，每元素表示某时刻的某个跟踪目标，
%              需包含字段 Time、TrackID、Position（[x y] 或 [x y z]）。
%     truths - 结构体数组，每元素表示某时刻的某个真值目标，
%              需包含字段 Time、TruthID、Position。
%     varargin - （可选）Name-Value 参数：
%       DistanceScale - 距离尺度（米）。相似度 = max(0, 1 - 距离/尺度)。
%                       默认 2，与 trackCLEARMetrics 的 EuclideanScale 一致。
%       Alpha         - 定位相似度阈值序列，默认 0.05:0.05:0.95。
%
%   输出：
%     hotaTable - 表，含 HOTA(%)、DetA(%)、AssA(%)、LocA(%) 四项。
%
%   说明：
%     - 定位相似度采用线性距离相似度（仅用中心点，取 x、y 平面距离）。
%     - HOTA = mean_alpha( sqrt(DetA_alpha * AssA_alpha) )。
%     - DetA 为检测精度，AssA 为关联精度，LocA 为定位精度。

    % 解析可选参数
    parser = inputParser;
    addParameter(parser, 'DistanceScale', 2, @(x) isscalar(x) && x > 0);
    addParameter(parser, 'Alpha', 0.05:0.05:0.95);
    parse(parser, varargin{:});
    scale = parser.Results.DistanceScale;
    alphas = parser.Results.Alpha;

    % 统一为列向量
    tracks = tracks(:);
    truths = truths(:);

    % 边界情况：无真值或无轨迹
    if isempty(truths) || isempty(tracks)
        hotaTable = table(0, 0, 0, 0, ...
            'VariableNames', {'HOTA(%)', 'DetA(%)', 'AssA(%)', 'LocA(%)'});
        return;
    end

    truthTimes = [truths.Time];
    trackTimes = [tracks.Time];
    timesteps = unique(truthTimes);

    % 提取平面位置（只取 x、y，忽略 z）。
    % 注意：Position 可能是行向量 [x y z] 或列向量 [x; y; z]，
    % 这里统一 reshape 成行向量后再垂直拼接，兼容两种写法。
    posCell = arrayfun(@(x) reshape(x.Position, 1, []), truths, 'UniformOutput', false);
    truthPos = vertcat(posCell{:});
    posCell = arrayfun(@(x) reshape(x.Position, 1, []), tracks, 'UniformOutput', false);
    trackPos = vertcat(posCell{:});
    nxy = min(2, size(truthPos, 2));
    truthPos = truthPos(:, 1:nxy);
    trackPos = trackPos(:, 1:nxy);

    truthIDs = [truths.TruthID];
    trackIDs = [tracks.TrackID];
    allTruthIDs = unique(truthIDs);
    allTrackIDs = unique(trackIDs);
    nTruthIDs = numel(allTruthIDs);
    nTrackIDs = numel(allTrackIDs);

    % 边界情况：无真值轨迹或无跟踪轨迹
    if nTruthIDs == 0 || nTrackIDs == 0 || isempty(timesteps)
        hotaTable = table(0, 0, 0, 0, ...
            'VariableNames', {'HOTA(%)', 'DetA(%)', 'AssA(%)', 'LocA(%)'});
        return;
    end

    % ID -> 索引 的映射，加速内层查找
    truthID2idx = containers.Map(allTruthIDs, 1:nTruthIDs);
    trackID2idx = containers.Map(allTrackIDs, 1:nTrackIDs);

    % 每个真值 ID / 轨迹 ID 的出现帧数（用于 AssA 的 FNA/FPA）
    gtCount = zeros(1, nTruthIDs);
    for i = 1:nTruthIDs
        gtCount(i) = sum(truthIDs == allTruthIDs(i));
    end
    trCount = zeros(1, nTrackIDs);
    for i = 1:nTrackIDs
        trCount(i) = sum(trackIDs == allTrackIDs(i));
    end

    nAlpha = numel(alphas);
    DetA = zeros(1, nAlpha);
    AssA = zeros(1, nAlpha);
    totalSim = 0;    % 累计定位相似度（用于 LocA）
    totalTP = 0;     % 累计 TP 次数（用于 LocA）

    for a = 1:nAlpha
        alpha = alphas(a);
        % potentialMatches(g, p)：真值轨迹 g 与跟踪轨迹 p 的匹配帧数
        potentialMatches = zeros(nTruthIDs, nTrackIDs);
        TP = 0; FP = 0; FN = 0;

        for t = 1:numel(timesteps)
            tt = timesteps(t);
            truthIdx = find(truthTimes == tt);
            trackIdx = find(trackTimes == tt);

            numT = numel(truthIdx);
            numK = numel(trackIdx);
            if numT == 0
                FP = FP + numK;
                continue;
            elseif numK == 0
                FN = FN + numT;
                continue;fprintf('===== 路口 %d 车辆 CLEAR 指标 =====\n', juncNum);
    disp(vehiclemetricTable);
    fprintf('===== 路口 %d 车辆 HOTA 指标 =====\n', juncNum);
            end

            % 相似度矩阵（真值 x 轨迹）
            S = localSimilarity(truthPos(truthIdx, :), trackPos(trackIdx, :), scale);
            Scut = S;
            Scut(S < alpha) = 0;

            % 每帧匹配：最大化相似度（匈牙利算法）
            matching = matchpairs(Scut, 0, 'max');
            numMatch = size(matching, 1);fprintf('===== 路口 %d 车辆 CLEAR 指标 =====\n', juncNum);
    disp(vehiclemetricTable);
    fprintf('===== 路口 %d 车辆 HOTA 指标 =====\n', juncNum);

            TP = TP + numMatch;
            FN = FN + numT - numMatch;
            FP = FP + numK - numMatch;

            for m = 1:numMatch
                ti = matching(m, 1);
                pi = matching(m, 2);
                gi = truthID2idx(truthIDs(truthIdx(ti)));
                pj = trackID2idx(trackIDs(trackIdx(pi)));
                potentialMatches(gi, pj) = potentialMatches(gi, pj) + 1;
                totalSim = totalSim + S(ti, pi);
                totalTP = totalTP + 1;
            end
        end

        % 检测精度 DetA
        DetA(a) = TP / (TP + FP + FN);

        % 全局关联匹配：最大化真值-轨迹的总匹配帧数
        globalMatching = matchpairs(-potentialMatches, 0);

        % 关联精度 AssA
        assASum = 0;
        for m = 1:size(globalMatching, 1)
            gi = globalMatching(m, 1);
            pj = globalMatching(m, 2);
            tpa = potentialMatches(gi, pj);
            if tpa == 0
                continue;
            end
            fna = gtCount(gi) - tpa;   % 真值 g 未与主匹配轨迹关联的帧数
            fpa = trCount(pj) - tpa;   % 轨迹 p 未与主匹配真值关联的帧数
            assASum = assASum + tpa / (tpa + fna + fpa);
        end
        AssA(a) = assASum / nTruthIDs;
    end

    % 在定位阈值上取平均
    HOTA = mean(sqrt(DetA .* AssA));
    DetA_m = mean(DetA);
    AssA_m = mean(AssA);
    LocA = totalSim / max(1, totalTP);

    hotaTable = table(100*HOTA, 100*DetA_m, 100*AssA_m, 100*LocA, ...
        'VariableNames', {'HOTA(%)', 'DetA(%)', 'AssA(%)', 'LocA(%)'});
end

function S = localSimilarity(posA, posB, scale)
%localSimilarity 计算两组平面位置之间的线性距离相似度
%   S(i,j) = max(0, 1 - ||posA(i,:) - posB(j,:)|| / scale)
    nA = size(posA, 1);
    nB = size(posB, 1);
    S = zeros(nA, nB);
    for i = 1:nA
        for j = 1:nB
            S(i, j) = max(0, 1 - norm(posA(i, :) - posB(j, :)) / scale);
        end
    end
end
