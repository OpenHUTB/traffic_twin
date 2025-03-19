classdef trackCLEARMetrics < handle & matlab.mixin.CustomDisplay
    %trackCLEARMetrics CLEAR Multi-Object Tracking metrics
    %
    % tcm = trackCLEARMetrics creates a CLEAR metrics object. Use this
    % object to evaluate the Classification of Events, Activities and
    % Relationships (CLEAR) Multi-Object Tracking metrics.
    %
    % tcm = trackCLEARMetrics("Name", value) additionally lets you
    % specify Name-Value properties.
    %
    % trackCLEARMetrics properties:
    %  SimilarityThreshold  - Threshold to assign track to truth
    %  SimilarityMethod     - Choice of similarity method (Construction only)
    %  EuclideanScale       - Euclidean scale for Euclidean Similarity (*)
    %  CustomSimilarityFcn  - Custom similarity function (Construction only) (**)
    %
    %  (*) Applicable only when SimilarityMethod = "Euclidean"
    %  (**)Applicable only when SimilarityMethod = "Custom"
    %
    % trackCLEARMetrics object function:
    %  evaluate             - Evaluate a sequence of tracks and truths
    %
    % Example:
    % % Create a CLEAR object
    % tcm = trackCLEARMetrics(SimilarityThreshold = 0.8);
    % % Load vision tracking dataset consisting of true objects (truths) and
    % % estimated objects (tracks)
    % load('trackCLEARData.mat','tracks','truths');
    % % Visualize tracks in red and truths in blue
    % figure
    % for t=1:15
    %     tracks_t = tracks([tracks.Time] == t);
    %     truths_t = truths([truths.Time] == t);
    %     for j=1:numel(tracks_t)
    %         rectangle('Position',tracks_t(j).BoundingBox,'EdgeColor','r','Curvature',[0.7 1]);
    %     end
    %     for j=1:numel(truths_t)
    %         rectangle('Position',truths_t(j).BoundingBox,'EdgeColor','b');
    %     end
    %     pause(1)
    % end
    % % Evaluate CLEAR MOT Metrics
    % clearTable = evaluate(tcm, tracks, truths)
    %
    %See also: trackAssignmentMetrics, trackErrorMetrics, trackOSPAMetric, trackGOSPAMetric

    % References:
    % [1] Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple
    % object tracking performance: the clear mot metrics." EURASIP Journal
    % on Image and Video Processing 2008 (2008): 1-10.
    % [2] Li, Yuan, Chang Huang, and Ram Nevatia. "Learning to associate:
    % Hybridboosted multi-target tracker for crowded scene." In 2009 IEEE
    % conference on computer vision and pattern recognition, pp. 2953-2960.
    % IEEE, 2009.

    % Copyright 2022 The MathWorks, Inc.


    properties
        %SimilarityThreshold Threshold for assigning track to truth
        % A true positive match between a track and truth is possible if
        % and only if the similarity between the track and the truth is
        % greater than or equal to SimilarityThreshold.
        % Default: 0.5
        SimilarityThreshold (1,1) {mustBeNumeric, mustBeNonnegative, mustBeFinite}   = 0.5

        %EuclideanScale Scale for Euclidean Similarity
        % When the property SimiliarityMethod is set to "Euclidean" the
        % similarity between a truth and a track is calculated as:
        %     max(0, 1 - EuclideanDistance / EuclideanScale)
        % where EuclideanDistance is the Euclidean distance between
        % truth.Position and track.Position vectors. Specify EuclideanScale
        % as a positive numeric scalar.
        % Default: 1
        EuclideanScale (1,1) {mustBeNumeric, mustBePositive} = 1
    end

    properties (SetAccess = immutable)
        %SimilarityMethod Choice of similarity method (Construction only)
        % Specify the method used to calculate the similarity between
        % tracks and truths. The choice of similarity determines the format
        % of tracks and truths inputs. The available options are:
        %  "IoU2d"     - The similarity is Intersection over Union
        %                for 2D axis-aligned rectangles. Tracks and truths
        %                inputs must define a "BoundingBox" field to
        %                specify the bounding box pixel coordinates as [x y
        %                width height], x and y are the top left corner
        %                pixel coordinates. (Default)
        % "Euclidean"  - The similarity is based on the Euclidean distance
        %                between track and truth. Tracks and truths inputs
        %                must define a "Position" field to specify their
        %                position vector. When using "Euclidean", refer to
        %                the object property <a href="matlab:help('trackCLEARMetrics/EuclideanScale')">EuclideanScale</a> for more
        %                information.
        % "Custom"     - The similarity is provided with the property
        %                <a href="matlab:help('trackCLEARMetrics/CustomSimilarityFcn')">CustomSimilarityFcn</a> as a function handle. Refer to
        %                the help of this property for more information.
        SimilarityMethod (1,1) fusion.internal.metrics.SimilarityEnum

        %CustomSimilarityFcn Custom similarity function (Construction only)
        % This property is only used when the property SimilarityMethod is
        % set to "Custom". Specify a function handle to calculate a custom
        % similarity value between tracks and truths. The function must
        % satisfy the following signature:
        %            similarity = customFcn(tracks, truths)
        % It must support arrays of track structures and truth structures.
        % The output similarity shall be an N-by-M matrix of scalars
        % between 0 and 1. N is the number of elements in tracks, and M is
        % the number of elements in truths, and similarity(i,j) is the
        % similarity between tracks(i) and truths(j).
        CustomSimilarityFcn function_handle {mustBeScalarOrEmpty}
    end

    properties (Access = protected)
        TableVarNames = {'MOTA (%)','MOTP (%)','Mostly Tracked (%)',...
            'Partially Tracked (%)','Mostly Lost (%)','False Positive','False Negative',...
            'Recall (%)','Precision (%)','False Track Rate',...
            'ID Switches','Fragmentations'}
    end

    properties(Access = private)
        pSimilarityFcn
    end

    methods
        function obj = trackCLEARMetrics(varargin)

            % Parse inputs
            parser = inputParser;
            addParameter(parser, 'SimilarityThreshold',obj.SimilarityThreshold);
            addParameter(parser, 'SimilarityMethod', 'IoU2d');
            addParameter(parser, 'CustomSimilarityFcn',function_handle.empty);
            addParameter(parser, 'EuclideanScale', obj.EuclideanScale);
            parse(parser, varargin{:});
            obj.SimilarityThreshold = parser.Results.SimilarityThreshold;
            obj.SimilarityMethod = parser.Results.SimilarityMethod;
            obj.CustomSimilarityFcn = parser.Results.CustomSimilarityFcn;
            obj.EuclideanScale = parser.Results.EuclideanScale;

            % Set Similarity Function from choice of method
            setSimilarityFcn(obj);
        end

        function resultsTable = evaluate(obj, tracks, truths)
            %evaluate Evaluate CLEAR metrics for a sequence of tracks and truths
            % results = evaluate(obj, tracks, truths) returns the CLEAR metrics for
            % the sequence of tracks and truths.
            % Inputs:
            %
            %        tracks   - An array of structs. Each element is a struct
            %                   representing the estimate of an object at a
            %                   timestep. The track struct must contains
            %                   the fields "Time" representing the time in
            %                   the sequence, and the field "TrackID"
            %                   representing the unique track identifier.
            %                   Additional fields are required depending on
            %                   the property SimilarityMethod. Refer to the
            %                   description of that property for more
            %                   information.
            %
            %        truths   - An array of structs. Each element is a struct
            %                   representing the true object at a timestep.
            %                   The track struct must contains the fields
            %                   "Time" representing the time in the
            %                   sequence, and the field "TruthID"
            %                   representing the unique truth identifier.
            %                   Additional fields are required depending on
            %                   the property SimilarityMethod. Refer to the
            %                   description of that property for more
            %                   information.
            %
            % Outputs:
            %        results is a table with CLEAR, Mostly-Tracked,
            %        Partially-Tracked, and Mostly-Lost MOT metrics value
            %        in the following order:
            %
            %        - MOTA              Multiple Object Tracking Accuracy
            %                            a percentage. Higher is better.
            %        - MOTP              Multiple Object Tracking Precision
            %                            a percentage. Higher is better.
            %        - Mostly Tracked    Percentage of true trajectories tracked 
            %                            more than 80% of their lifetime.
            %                            Higher is better.
            %        - Partially Tracked Percentage of true trajectories tracked 
            %                            more less than 80% but more than
            %                            20% of their lifetime.
            %        - Mostly Lost       Percentage of true trajectories tracked
            %                            less than 20% of their lifetime.
            %                            Lower is better.
            %        - False Positive    Total number of tracks that are not
            %                            matched with any true object.
            %                            Lower is better.
            %        - False Negative    Total Number of truths that are not
            %                            matched with any estimated object.
            %                            Lower is better.
            %        - Recall            Percentage of true objects being
            %                            tracked. Higher is better.
            %        - Precision         Percentage of estimated objects
            %                            matching true object. Higher is
            %                            better.
            %        - False Track Rate  Average number of false tracks, 
            %                            or false positives per timestep
            %                            (frame). Lower is better.
            %        - ID Switches       Total number of Identity switches. 
            %                            An identity switch occurs when a
            %                            true object is tracked at time t
            %                            and is tracked by a different
            %                            track at time t+1. Lower is
            %                            better.
            %        - Fragmentations    Total number of Fragmentations. A
            %                            fragmentation occurs when a true
            %                            object is tracked again, after
            %                            being untracked for at least one
            %                            timestep. Lower is better.


            % Input parsing and validation
            narginchk(3,3);
            [nsteps, tracksPerStep, truthsPerStep, alltruthIDs] = validateEvalInputs(obj, tracks, truths);

            % Initialize results struct
            ntruths = numel(alltruthIDs);
            results = initializeResults(obj, nsteps, ntruths);

            % Initialize Matches
            Mprev = [];

            for t = 1:nsteps
                curTruths = truthsPerStep{t};
                curTracks = tracksPerStep{t};
                numTruths = numel(curTruths);
                numTracks = numel(curTracks);

                if numTruths == 0
                    results.FP = results.FP + numTracks;
                    continue
                elseif numTracks == 0
                    results.FN = results.FN + numTruths;
                    continue
                end

                trackids = [curTracks.TrackID]';
                truthids = [curTruths.TruthID]';

                % calculate similarity between all truth and track pairs
                similarityScore = obj.pSimilarityFcn(curTracks, curTruths);
                score = similarityScore;
                if ~isempty(Mprev) % Preserve previous matches if possible
                    [~,tmp1,~] = intersect(Mprev(:,1), trackids,'stable');
                    [~,tmp2,prevTruthInd] = intersect(Mprev(tmp1,2),truthids, 'stable');
                    [~,~,prevTrackInd] = intersect(Mprev(tmp1(tmp2),1), trackids,'stable');
                    linind = sub2ind(size(similarityScore),prevTrackInd,prevTruthInd);
                    score(linind) = score(linind) + realmax;
                end

                % Apply Threshold
                score (similarityScore < obj.SimilarityThreshold) = 0;

                % Find matching that maximize the similarity
                matching = matchpairs(score, 0, 'max');

                % Map to trackID and truthID
                M = [trackids(matching(:,1)), truthids(matching(:,2))];

                % Aggregate metrics
                switches = calculateSwitches(obj,Mprev, M);
                results.IDs = results.IDs + switches ;
                numMatches = size(matching, 1);
                results.TP = results.TP + numMatches ;
                results.FN = results.FN + numTruths - numMatches;
                results.FP  = results.FP + numTracks - numMatches;

                if numMatches > 0
                    % Extract similarity score of the matchings
                    allscores = similarityScore(sub2ind(size(similarityScore),matching(:,1),matching(:,2)));
                    results.cumsimilarity = results.cumsimilarity +sum(allscores);
                end

                % Increment counters
                if ~isempty(truthids)
                    [~,truth2incr] = ismember(truthids, alltruthIDs);
                    results.truthCounter(truth2incr) = results.truthCounter(truth2incr) +1;
                end
                if ~isempty(M)
                    [~,match2incr] = ismember(M(:,2), alltruthIDs);
                    results.matchCounter(match2incr) = results.matchCounter(match2incr) +1;
                    % Increment Fragmentation counter
                    if isempty(Mprev)
                        fragTruthIDs = M(:,2);
                    else
                        [~,ind] = setdiff(M(:,2), Mprev(:,2));
                        fragTruthIDs = M(ind,2);
                    end
                    [~,frag2incr] = ismember(fragTruthIDs, alltruthIDs);
                    results.fragCounter(frag2incr) = results.fragCounter(frag2incr) + 1;
                end

                % Save matches for next timestep
                Mprev = M;
            end

            % Calculate derived metrics
            results = calcDerivedMetrics(obj, results);

            % format results in a table
            resultsTable = createResultsTable(obj, results);
        end
    end

    methods(Access = protected)

        function nswitch = calculateSwitches(~,prevMatch, curMatch)
            %Each track assignment from prevMatch that no longer occurs in
            %curMatch is an Identity Switch
            nswitch = 0;
            for i=1:size(prevMatch, 1)
                prevTrackID = prevMatch(i,1);
                prevTruthID = prevMatch(i,2);
                prevTruthInd = curMatch(:,2)==prevTruthID;
                if any(prevTruthInd) && (curMatch(prevTruthInd, 1) ~= prevTrackID)
                    nswitch = nswitch +1;
                end
            end
        end

        function results = initializeResults(~, nsteps, ntruths)
            % Initialize sub-metrics to aggregate over the sequence

            results = struct('NumSteps',nsteps,...
                'NumTruths', ntruths,...
                'FP', 0,...
                'FN', 0 ,...
                'IDs', 0, ...
                'TP', 0, ...
                'cumsimilarity', 0,...
                'matchCounter', zeros(1,ntruths), ... % for each true object count number of timesteps present
                'truthCounter', zeros(1,ntruths),...  % for each true object, count number of matches
                'fragCounter', zeros(1, ntruths));
        end

        function results = calcDerivedMetrics(~, results)
            % The following metrics are derived based on aggregated
            % sub-metrics: MOTA, Recall, Precision
            results.MOTA = (results.TP - results.FP - results.IDs)/max(1, results.TP + results.FN);
            results.MOTP = results.cumsimilarity / max(1, results.TP);
            results.Precision = results.TP / max(1, results.TP + results.FP);
            results.Recall = results.TP / max(1, results.TP + results.FN);

            % Calculate ratio for MT, ML, PT
            % Calculate MT Ml, PT, from counters
            trackedRatio = results.matchCounter(results.truthCounter  >0) ./ results.truthCounter(results.truthCounter>0);
            results.MT = sum(trackedRatio > 0.8);
            results.PT = sum(trackedRatio >= 0.2) - results.MT;
            results.ML = results.NumTruths - results.MT - results.PT;
            % Return ratio
            results.MT = results.MT / results.NumTruths;
            results.PT = results.PT / results.NumTruths;
            results.ML = results.ML / results.NumTruths;

            %Fragmentations
            results.Frag = sum(results.fragCounter(results.fragCounter>0) - 1);
            % Calculate False Track Ratio
            results.FTR = results.FP / max(1, results.NumSteps);

        end

        function resultsTable = createResultsTable(obj, results)
            % format results in a table
            resultsTable = table(100*results.MOTA,...
                100*results.MOTP,...
                100*results.MT,...
                100*results.PT,...
                100*results.ML,...
                results.FP,...
                results.FN,...
                100*results.Recall,...
                100*results.Precision,...
                results.FTR,...
                results.IDs,...
                results.Frag,...
                'VariableNames', obj.TableVarNames);
        end

        function [nsteps, tracksPerStep, truthsPerStep, alltruthIDs] = validateEvalInputs(obj, tracksIn, truthsIn)

            % Verify tracks and truths are both structs
            tracks = tracksIn(:);
            truths = truthsIn(:);

            switch obj.SimilarityMethod
                case "IoU2d"
                    reqTrackFields = {'Time','TrackID','BoundingBox'};
                    reqTruthFields = {'Time','TruthID','BoundingBox'};
                case "Euclidean"
                    reqTrackFields = {'Time','TrackID','Position'};
                    reqTruthFields = {'Time','TruthID','Position'};
                case "Custom"
                    % Exercise SimilarityFcn for validation
                    mustBeNonempty(tracks);
                    mustBeNonempty(truths);
                    try
                        obj.pSimilarityFcn(tracks(1), truths(1));
                    catch ME
                        throwAsCaller(ME);
                    end
            end

            if obj.SimilarityMethod ~= "Custom"
                validateattributes(tracks, {'struct'},{'vector','nonempty'});
                validateattributes(truths, {'struct'},{'vector','nonempty'});
                isfieldmissing = ~isfield(tracks,reqTrackFields);
                if any(isfieldmissing)
                    missingfield = reqTrackFields(isfieldmissing);
                    error(message('fusion:trackCLEARMetrics:BadIoUTrackFormat',"SimilarityMethod",string(obj.SimilarityMethod),missingfield{1}));
                end
                isfieldmissing = ~isfield(truths,reqTruthFields);
                if any(isfieldmissing)
                    missingfield = reqTruthFields(isfieldmissing);
                    error(message('fusion:trackCLEARMetrics:BadIoUTruthFormat',"SimilarityMethod",string(obj.SimilarityMethod),missingfield{1}));
                end
            end

            % Parse truths to retrieve timestamps
            truthtimesteps = [truths.Time];
            timesteps = unique(truthtimesteps);
            nsteps = numel(timesteps);
            
            % Verify tracks timesteps is a subset of truths timesteps
            tracktimesteps = [tracks.Time];
            uniquetracktimesteps = unique(tracktimesteps);
            cond = sum(ismembertol(uniquetracktimesteps, timesteps)) == numel(uniquetracktimesteps);
            
            if ~cond
                % 如果时间戳不匹配，过滤掉多余的时间戳
                validIdx = ismembertol(tracktimesteps, timesteps);
                tracks = tracks(validIdx);
                tracktimesteps = [tracks.Time]; % 更新过滤后的时间戳
                uniquetracktimesteps = unique(tracktimesteps);
                cond = sum(ismembertol(uniquetracktimesteps, timesteps)) == numel(uniquetracktimesteps);
            end
            
            try
                assert(cond,message('fusion:trackCLEARMetrics:MismatchTimestamp','Time'));
            catch ME
                throwAsCaller(ME);
            end
 

            %Save all IDs
            alltruthIDs = unique([truths.TruthID]);

            % reformat tracks and truths
            tracksPerStep = cell(1,nsteps);
            truthsPerStep = cell(1, nsteps);
            for i=1:nsteps
                % find truths and tracks at the current timestep
                curtime = timesteps(i);
                truthsPerStep{i} = truths(truthtimesteps == curtime);
                tracksPerStep{i} = tracks(tracktimesteps == curtime);
            end
        end

        function setSimilarityFcn(obj)
            % Set function from options
            switch obj.SimilarityMethod
                case "IoU2d"
                    obj.pSimilarityFcn =  @fusion.internal.metrics.similarityIoU;
                case "Euclidean"
                    obj.pSimilarityFcn = @(x,y) fusion.internal.metrics.similarityEuclidean(x,y,obj.EuclideanScale);
                otherwise
                    obj.pSimilarityFcn = obj.CustomSimilarityFcn;
            end

            % Warn if a custom function is specified but not used
            if obj.SimilarityMethod ~= "Custom" && ~isempty(obj.CustomSimilarityFcn)
                warning(message('fusion:trackCLEARMetrics:CustomFcnNotUsed', 'CustomSimilarityFcn'));
            end
        end
    end

    methods (Access='protected')
        function groups = getPropertyGroups(obj)
            % Define property section(s) for display in Matlab
            if obj.SimilarityMethod == "Euclidean"
                groups = matlab.mixin.util.PropertyGroup({'SimilarityThreshold',...
                'SimilarityMethod','EuclideanScale'});
            elseif obj.SimilarityMethod == "Custom"
                groups = matlab.mixin.util.PropertyGroup({'SimilarityThreshold',...
                'SimilarityMethod','CustomSimilarityFcn'});
            else
                groups = matlab.mixin.util.PropertyGroup({'SimilarityThreshold',...
                'SimilarityMethod'});
            end
        end
    end
end
