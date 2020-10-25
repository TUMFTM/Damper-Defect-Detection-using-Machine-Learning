function evaluatePredictions(classifier,featureStruct,varargin)
%EVALUATECLASSIFIERPREDICTIONS Summary of this function goes here
%   Detailed explanation goes here

    ctrl.plotGeneralDataset = 0;
    ctrl.furtherAnalysis = 1;

    edgesForHistogram = [0:0.05:1];
    
    [predictedClass, NegLoss, PBScore, Posterior] = predict(classifier, table2array(featureStruct.data));
    
    fieldsDataProp = fields(featureStruct.Prop);
    observation_properties = featureStruct.Prop;
    
    TrueClassPosterior = Posterior.*featureStruct.labelAsMatrix;
    TrueClassPosterior(TrueClassPosterior==0) = NaN;    % change 0 with NaN for histogram function
    [numberOfTrueClassPosteriorsForHistogram,~] = histcounts(TrueClassPosterior,edgesForHistogram);
    
    % Convert predictedClass labels to matrix representation
    predictedClassAsMatrix = zeros(size(predictedClass,1),size(classifier.ClassNames,1));
    for cntUniqueClasses = 1 : size(classifier.ClassNames,1)
        predictedClassAsMatrix(:,cntUniqueClasses) = strcmp(classifier.ClassNames{cntUniqueClasses},predictedClass);
    end
    
    PredictedClassPosterior = Posterior.*predictedClassAsMatrix;
    PredictedClassPosterior(PredictedClassPosterior==0) = NaN;    % change 0 with NaN for histogram function
    [numberOfPredictedClassPosteriorsForHistogram,~] = histcounts(PredictedClassPosterior,edgesForHistogram);
    
    figure;
    hold on
    plot(edgesForHistogram(2:end),numberOfTrueClassPosteriorsForHistogram,'DisplayName','TrueClass')
    plot(edgesForHistogram(2:end),numberOfPredictedClassPosteriorsForHistogram,'DisplayName','PredictedClass')
    xlabel('Posterior Probability');
    ylabel('Number of Observations');
    legend show
    
    figure;
    numberOfTrueClassPosteriorsForHistogramForEachClass = zeros(length(edgesForHistogram)-1,size(TrueClassPosterior,2));
    numberOfPredictedClassPosteriorsForHistogramForEachClass = zeros(length(edgesForHistogram)-1,size(PredictedClassPosterior,2));
    for cntClass = 1 : size(classifier.ClassNames,1)
        subplot(2,2,cntClass);
        hold on
        
        % True Class
        [numberOfTrueClassPosteriorsForHistogramForEachClass(:,cntClass),~] = ...
            histcounts(TrueClassPosterior(:,cntClass),[0:0.05:1]);
        plot(edgesForHistogram(2:end),numberOfTrueClassPosteriorsForHistogramForEachClass(:,cntClass),'DisplayName','True Class')
        
        % Predicted Class
        [numberOfPredictedClassPosteriorsForHistogramForEachClass(:,cntClass),~] = ...
            histcounts(PredictedClassPosterior(:,cntClass),[0:0.05:1]);
        plot(edgesForHistogram(2:end),numberOfPredictedClassPosteriorsForHistogramForEachClass(:,cntClass),'DisplayName','Predicted Class')
        
        xlabel('Posterior Probability');
        ylabel('Number of Observations');
        legend show
        title(classifier.ClassNames{cntClass});
    end
    
    
%     if sum(tf) == size(featureStruct,1)
%         tmptf = tf;
%     elseif sum(~tf) == size(featureStruct,1)
%         tmptf = ~tf;
%     else
%         fprintf('Wrong tf-Variable!\n');
%     end
%     for cntDataPropFields = 1 : length(fieldsDataProp)
%         try
%             observation_properties.(fieldsDataProp{cntDataPropFields}) = data.Prop.(fieldsDataProp{cntDataPropFields})(tmptf,:);
%         catch
%             tmp = data.Prop.(fieldsDataProp{cntDataPropFields})(tmptf(1:size(data.Prop.(fieldsDataProp{cntDataPropFields}),1),:));
%             tmp2 = cell(size(observation_properties.(fieldsDataProp{1}),1)-size(tmp,1),1);
%             tmp2(:) = {'-'};
%             observation_properties.(fieldsDataProp{cntDataPropFields}) = [tmp; tmp2];
%         end
%     end
    
    %% Get data
    min_score = 0.2;

%     if exist('score') ==1 && isfield(observation_properties,'score') == 0 
%         observation_properties = [observation_properties table(score)];
%     end
% 
%     if exist('validation') ==1 && isfield(observation_properties,'validation') == 0 
%         observation_properties = [observation_properties table(validation)];
%     end

    %% General dataset plots
    if ctrl.plotGeneralDataset
        
        set(figure,'Name','General dataset and track information','NumberTitle','off');
        subplot(2,2,1);
        cat_detection = categorical(observation_properties.labelDetection);
        histogram(cat_detection);
        title('Detection classes')
        ylabel('number of observations')

        subplot(2,2,2);
        cat_isolation = categorical(observation_properties.labelIsolation);
        histogram(cat_isolation);
        title('Isolation & detection classes')
        ylabel('number of observations')

        subplot(2,2,3);
        cat_track = categorical(observation_properties.track);
        histogram(cat_track);
        title('Tracks')
        ylabel('number of observations')

        subplot(2,2,4);
        cat_track_condition = categorical(observation_properties.trackCondition);
        histogram(cat_track_condition);
        title('Track conditions');
        ylabel('number of observations')
        
    end

    %% Isolated dataset information | high vs. low confidence level
    
    if ctrl.furtherAnalysis
    
        set(figure,'Name','Comparison of high and low confidence level data','NumberTitle','off');

        subplot(2,2,1);
        h = [];
        if isfield(featureStruct.Prop,'here')
            % IRI Analysis -- hier muss noch was getan werden
            [~, edgesIRI, binIRI]= histcounts(featureStruct.Prop.here.AvgIRI,5);
            uniqueBinIRI = unique(binIRI);
            uniqueBinIRI(uniqueBinIRI==0) = [];
            for cntBinIRI = 1 : length(uniqueBinIRI)
                relevantObservationsIdx = (binIRI==uniqueBinIRI(cntBinIRI));
                relevantObservationPosterior = Posterior(relevantObservationsIdx,:);
                relevantTrueClass = observation_properties.labelIsolation(relevantObservationsIdx,:);
                relevantTrueClassPosterior = zeros(size(relevantObservationPosterior,1),1);
                for cntRelevantObservations = 1 : size(relevantObservationPosterior,1)
                    trueClassIdx = contains(classifier.ClassNames,relevantTrueClass{cntRelevantObservations});
                    relevantTrueClassPosterior(cntRelevantObservations,:) = relevantObservationPosterior(cntRelevantObservations, trueClassIdx);
                end
                [counts, bin] = hist(relevantTrueClassPosterior,[0:0.05:1]);
                h(end+1) = plot(bin, counts, 'DisplayName', [num2str(edgesIRI(cntBinIRI)), ' <= IRI < ', num2str(edgesIRI(cntBinIRI+1))]);
                hold on
            end
            legend show
            title('IRI Analysis');
            xlabel('True Class Posterior Probability');
            ylabel('Number of Observations')
            
        else
            % Track Analysis
            uniqueTracks = unique(observation_properties.track);
            for cntTracks = 1 : length(uniqueTracks)
                relevantObservationsIdx = strcmp(uniqueTracks{cntTracks},observation_properties.track);
                relevantObservationPosterior = Posterior(relevantObservationsIdx,:);
                relevantPredictedClass = predictedClass(relevantObservationsIdx,:);
                relevantTrueClass = observation_properties.labelIsolation(relevantObservationsIdx,:);
                relevantTrueClassPosterior = zeros(size(relevantObservationPosterior,1),1);
                for cntRelevantObservations = 1 : size(relevantObservationPosterior,1)
                    trueClassIdx = contains(classifier.ClassNames,relevantTrueClass{cntRelevantObservations});
                    relevantTrueClassPosterior(cntRelevantObservations,:) = relevantObservationPosterior(cntRelevantObservations, trueClassIdx);
                end
                [counts, bin] = hist(relevantTrueClassPosterior,[0:0.05:1]);
                h(end+1) = plot(bin, counts);
                hold on
            end
            legend(h,uniqueTracks);
            title('Track Analysis');
            xlabel('True Class Posterior Probability');
            ylabel('Number of Observations')
        end

        % Track Condition Analysis
        subplot(2,2,2);
        h = [];
        
        if isfield(featureStruct.Prop,'here')
            uniqueTrackCondition = unique(featureStruct.Prop.here.RoughnessCat);
            uniqueTrackCondition(isnan(uniqueTrackCondition)) = [];
            trackCondition = featureStruct.Prop.here.RoughnessCat;
        else
            uniqueTrackCondition = unique(observation_properties.trackCondition);
            trackCondition = observation_properties.trackCondition;
        end
        for cntTracks = 1 : length(uniqueTrackCondition)
            if iscell(uniqueTrackCondition)
                relevantObservationsIdx = strcmp(uniqueTrackCondition{cntTracks},trackCondition);
            else
                relevantObservationsIdx = find(trackCondition==uniqueTrackCondition(cntTracks));
            end
            relevantObservationPosterior = Posterior(relevantObservationsIdx,:);
            relevantPredictedClass = predictedClass(relevantObservationsIdx,:);
            relevantTrueClass = observation_properties.labelIsolation(relevantObservationsIdx,:);
            relevantTrueClassPosterior = zeros(size(relevantObservationPosterior,1),1);
            for cntRelevantObservations = 1 : size(relevantObservationPosterior,1)
                trueClassIdx = contains(classifier.ClassNames,relevantTrueClass{cntRelevantObservations});
                relevantTrueClassPosterior(cntRelevantObservations,:) = relevantObservationPosterior(cntRelevantObservations, trueClassIdx);
            end
            [counts, bin] = hist(relevantTrueClassPosterior,[0:0.05:1]);
            h(end+1) = plot(bin, counts);
            hold on
        end
        if iscell(uniqueTrackCondition)
            legend(h,uniqueTrackCondition);
        else
            legend(h,num2str(uniqueTrackCondition));
        end
        title('Track Condition Analysis');
        xlabel('True Class Posterior Probability');
        ylabel('Number of Observations')

        % True Class Analysis
        subplot(2,2,3);
        h = [];
        uniqueClasses = unique(observation_properties.labelIsolation);
        for cntTracks = 1 : length(uniqueClasses)
            relevantObservationsIdx = strcmp(uniqueClasses{cntTracks},observation_properties.labelIsolation);
            relevantObservationPosterior = Posterior(relevantObservationsIdx,:);
            relevantPredictedClass = predictedClass(relevantObservationsIdx,:);
            relevantTrueClass = observation_properties.labelIsolation(relevantObservationsIdx,:);
            relevantTrueClassPosterior = zeros(size(relevantObservationPosterior,1),1);
            for cntRelevantObservations = 1 : size(relevantObservationPosterior,1)
                trueClassIdx = contains(classifier.ClassNames,relevantTrueClass{cntRelevantObservations});
                relevantTrueClassPosterior(cntRelevantObservations,:) = relevantObservationPosterior(cntRelevantObservations, trueClassIdx);
            end
            [counts, bin] = hist(relevantTrueClassPosterior,[0:0.05:1]);
            h(end+1) = plot(bin, counts);
            hold on
        end
        legend(h,uniqueClasses);
        title('True Class Analysis');
        xlabel('True Class Posterior Probability');
        ylabel('Number of Observations')

        
        % Track Analysis
        subplot(2,2,4);
        h = [];
        uniqueTracks = unique(observation_properties.track);
        for cntTracks = 1 : length(uniqueTracks)
            relevantObservationsIdx = strcmp(uniqueTracks{cntTracks},observation_properties.track);
            relevantObservationPosterior = Posterior(relevantObservationsIdx,:);
            relevantPredictedClass = predictedClass(relevantObservationsIdx,:);
            relevantTrueClass = observation_properties.labelIsolation(relevantObservationsIdx,:);
            relevantTrueClassPosterior = zeros(size(relevantObservationPosterior,1),1);
            for cntRelevantObservations = 1 : size(relevantObservationPosterior,1)
                trueClassIdx = contains(classifier.ClassNames,relevantTrueClass{cntRelevantObservations});
                relevantTrueClassPosterior(cntRelevantObservations,:) = relevantObservationPosterior(cntRelevantObservations, trueClassIdx);
            end
            [counts, bin] = hist(relevantTrueClassPosterior,[0:0.05:1]);
            h(end+1) = plot(bin, counts);
            hold on
        end
        legend(h,uniqueTracks);
        title('Track Analysis');
        xlabel('True Class Posterior Probability');
        ylabel('Number of Observations')
        
%         % Predicted Class Analysis
%         subplot(2,2,4);
%         h = [];
%         uniqueClasses = unique(observation_properties.labelIsolation);
%         for cntTracks = 1 : length(uniqueClasses)
%             relevantObservationsIdx = strcmp(uniqueClasses{cntTracks},observation_properties.labelIsolation);
%             relevantObservationPosterior = Posterior(relevantObservationsIdx,:);
%             relevantPredictedClass = predictedClass(relevantObservationsIdx,:);
%             relevantTrueClass = observation_properties.labelIsolation(relevantObservationsIdx,:);
%             relevantPredictedClassPosterior = zeros(size(relevantObservationPosterior,1),1);
%             for cntRelevantObservations = 1 : size(relevantObservationPosterior,1)
%                 predictedClassIdx = contains(classifier.ClassNames,relevantPredictedClass{cntRelevantObservations});
%                 relevantPredictedClassPosterior(cntRelevantObservations,:) = relevantObservationPosterior(cntRelevantObservations, predictedClassIdx);
%             end
%             [counts, bin] = hist(relevantPredictedClassPosterior,[0:0.05:1]);
%             h(end+1) = plot(bin, counts);
%             hold on
%         end
%         legend(h,uniqueClasses);
%         title('Predicted Class Analysis');
%         xlabel('Predicted Class Posterior Probability');
%         ylabel('Number of Observations')
        
    end

%     %% Isolated dataset information | right vs. wrong classification
%     rows = ~strcmp(observation_properties.labelIsolation,predictedClass);
% 
%     set(figure,'Name','Comparison of right and wrong classification data','NumberTitle','off');
%     subplot(1,3,1);
%     cat_track_iso = categorical(observation_properties.track,unique(observation_properties.track),'Ordinal',false);
%     h1 = histogram(cat_track_iso(rows));
%     h1.FaceColor = 'gr';
%     hold on
%     h2 = histogram(cat_track_iso(~rows));
%     h2.FaceColor = 'r';
%     title('Tracks')
%     ylabel('number of observations')
%     legend('right classification','wrong classification')
% 
%     subplot(1,3,2);
%     cat_track_condition_iso = categorical(observation_properties.trackCondition,unique(observation_properties.trackCondition),...
%         'Ordinal',false);
%     h1 = histogram(cat_track_condition_iso(rows));
%     h1.FaceColor = 'gr';
%     hold on
%     h2 = histogram(cat_track_condition_iso(~rows));
%     h2.FaceColor = 'r';
%     title('Track conditions')
%     ylabel('number of observations')
%     legend('right classification','wrong classification')
% 
%     subplot(1,3,3);
%     cat_labelIsolation_iso = categorical(observation_properties.labelIsolation,unique(observation_properties.labelIsolation),...
%         'Ordinal',false);
%     h1 = histogram(cat_labelIsolation_iso(rows));
%     h1.FaceColor = 'gr';
%     hold on
%     h2 = histogram(cat_labelIsolation_iso(~rows));
%     h2.FaceColor = 'r';
%     title('True Class')
%     ylabel('number of observations')
%     legend('right classification','wrong classification')
% 
%     subplot(2,2,2);
%     h1 = histogram(observation_properties.mean_acc_x(rows));
%     h1.BinWidth = 0.025;
%     h1.FaceColor = 'gr';
%     hold on
%     h2 = histogram(observation_properties.mean_acc_x(~rows));
%     h2.BinWidth = 0.025;
%     h2.FaceColor = 'r';
%     title('Mean acceleration_x');
%     xlabel('m/s^2')
%     ylabel('number of observations')
%     legend('right classification','wrong classification')
% 
%     subplot(2,2,3);
%     h1 = histogram(observation_properties.mean_acc_y(rows));
%     h1.BinWidth = 0.025;
%     h1.FaceColor = 'gr';
%     hold on
%     h2 = histogram(observation_properties.mean_acc_y(~rows));
%     h2.BinWidth = 0.025;
%     h2.FaceColor = 'r';
%     title('Mean acceleration_y');
%     xlabel('m/s^2')
%     ylabel('number of observations')
%     legend('right classification','wrong classification')


end

