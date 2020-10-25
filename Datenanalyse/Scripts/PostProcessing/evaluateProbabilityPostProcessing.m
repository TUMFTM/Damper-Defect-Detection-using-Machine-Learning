function probAnalysis = evaluateProbabilityPostProcessing(featuresTesting, featuresTraining, trainedClassifier)
%EVALUATEPROBABILITYPOSTPROCESSING Summary of this function goes here
%   Detailed explanation goes here

    fprintf('Starting Post-Processing');

    ctrl.plotConfusionMatrix = 1;
    ctrl.plotMeasurementAnalysis = 0;
    ctrl.plotROC = 0;
    ctrl.calcHiddenMarkovModel = 1;
    ctrl.calcSeriesPrediction = 0;

    if ctrl.plotROC
        handleROCfigure = figure;
        subplotRows = 2;
        subplotCols = 3;
        currSubplot = 0;
    end
    
    %% Generate TrueClass sequence
    if isstruct(featuresTesting)
        if isfield(featuresTesting, 'labelAsMatrix')
            tmpClassIdx = [1 : size(featuresTesting.labelAsMatrix,2)]';
            trueClassSequence = featuresTesting.labelAsMatrix * tmpClassIdx;
        end
    end
    
    %% Generate field 'batch'
    if ~isfield(featuresTesting.Prop, 'batch')
        featuresTesting.Prop.batch = featuresTesting.Prop.ID;
    end
    if ~isfield(featuresTraining.Prop, 'batch')
        featuresTraining.Prop.batch = featuresTraining.Prop.ID;
    end

    if ~iscell(trainedClassifier)
        tmpTrainedClassifier{1,1} = trainedClassifier;
    else
        tmpTrainedClassifier = trainedClassifier;
    end
    
    if ~iscell(featuresTesting.Prop.Posterior)
        tmpFeaturesTestingPosterior{1,1} = featuresTesting.Prop.Posterior;
    else
        tmpFeaturesTestingPosterior = featuresTesting.Prop.Posterior;
    end
    
    featuresTestingLabelAsMatrix = featuresTesting.labelAsMatrix;
    featuresTrainingLabelAsMatrix = featuresTraining.labelAsMatrix;
    if isfield(featuresTraining, 'data')
        featuresTrainingData = featuresTraining.data;
    end
    if isfield(featuresTraining.Prop, 'Posterior')
        featuresTrainingPosterior = featuresTraining.Prop.Posterior;
        if iscell(featuresTrainingPosterior)
            featuresTrainingPosterior = featuresTrainingPosterior{1,1};
        end
    end
    featuresTestingBatch = featuresTesting.Prop.batch;
    
    probAnalysisOut = cell(length(tmpFeaturesTestingPosterior),1);

    
    
    for cntCV = 1 : length(tmpFeaturesTestingPosterior)
        
        probAnalysis = struct();

        featuresTestingPosterior = tmpFeaturesTestingPosterior{cntCV};

        %% Apriori probability
        probAnalysis.apriori = struct();
        probAnalysis.apriori.Probabilities = featuresTestingPosterior;
        [~,probAnalysis.apriori.Sequence] = max(probAnalysis.apriori.Probabilities,[],2);

        % Calculate m-value (quasi multiclass AUC value)
        probAnalysis.apriori.mValue = multiClassAUC(probAnalysis.apriori.Probabilities,trueClassSequence);

        % Calculate accuracy
        [probAnalysis.apriori.misclassification,~,~,~] = confusion(featuresTestingLabelAsMatrix', featuresTestingPosterior');
        probAnalysis.apriori.accuracy = 1 - probAnalysis.apriori.misclassification;

        % Calculate AUC for each class
        probAnalysis.apriori.auc = zeros(1,size(featuresTestingLabelAsMatrix,2));
        for cntClass = 1 : size(featuresTestingLabelAsMatrix,2)
            [probAnalysis.apriori.auc(cntClass),~,~] = fastAUC(logical(featuresTestingLabelAsMatrix(:,cntClass)),probAnalysis.apriori.Probabilities(:,cntClass),ctrl.plotROC);
        end

        %% Mean probability
        probAnalysis.meanApriori = struct();

        % Calc probability
        [probAnalysis.meanApriori.Probabilities, probAnalysis.meanApriori.Sequence] = calcMeanProbabilityForEachMeasurement(featuresTestingPosterior, featuresTestingBatch);

        % Calculate m-value (quasi multiclass AUC value)
        probAnalysis.meanApriori.mValue = multiClassAUC(probAnalysis.meanApriori.Probabilities,trueClassSequence);

        % Calculate accuracy
        [probAnalysis.meanApriori.misclassification,~,~,~] = confusion(featuresTestingLabelAsMatrix', probAnalysis.meanApriori.Probabilities');
        probAnalysis.meanApriori.accuracy = 1 - probAnalysis.meanApriori.misclassification;

        % Calculate AUC for each class
        probAnalysis.meanApriori.auc = zeros(1,size(featuresTestingLabelAsMatrix,2));
        for cntClass = 1 : size(featuresTestingLabelAsMatrix,2)
            [probAnalysis.meanApriori.auc(cntClass),~,~] = fastAUC(logical(featuresTestingLabelAsMatrix(:,cntClass)),probAnalysis.meanApriori.Probabilities(:,cntClass),ctrl.plotROC);
        end

        %% Moving mean probability
        probAnalysis.movMeanApriori = struct();

        % Calc probability
        [probAnalysis.movMeanApriori.Probabilities, probAnalysis.movMeanApriori.Sequence] = calcMovMeanProbabilityForEachMeasurement(featuresTestingPosterior, featuresTestingBatch);

        % Calculate m-value (quasi multiclass AUC value)
        probAnalysis.movMeanApriori.mValue = multiClassAUC(probAnalysis.movMeanApriori.Probabilities,trueClassSequence);

        % Calculate accuracy
        [probAnalysis.movMeanApriori.misclassification,~,~,~] = confusion(featuresTestingLabelAsMatrix', probAnalysis.movMeanApriori.Probabilities');
        probAnalysis.movMeanApriori.accuracy = 1 - probAnalysis.movMeanApriori.misclassification;

        % Calculate AUC for each class
        probAnalysis.movMeanApriori.auc = zeros(1,size(featuresTestingLabelAsMatrix,2));
        for cntClass = 1 : size(featuresTestingLabelAsMatrix,2)
            [probAnalysis.movMeanApriori.auc(cntClass),~,~] = fastAUC(logical(featuresTestingLabelAsMatrix(:,cntClass)),probAnalysis.movMeanApriori.Probabilities(:,cntClass),ctrl.plotROC);
        end

        %% Weighted mean probability
        probAnalysis.weightedMeanApriori = struct();

        % Calc probability
        [probAnalysis.weightedMeanApriori.Probabilities, probAnalysis.weightedMeanApriori.Sequence] = calcWeightedMeanProbabilityForEachMeasurement(featuresTestingPosterior, featuresTestingBatch);

        % Calculate m-value (quasi multiclass AUC value)
        probAnalysis.weightedMeanApriori.mValue = multiClassAUC(probAnalysis.weightedMeanApriori.Probabilities,trueClassSequence);

        % Calculate accuracy
        [probAnalysis.weightedMeanApriori.misclassification,~,~,~] = confusion(featuresTestingLabelAsMatrix', probAnalysis.weightedMeanApriori.Probabilities');
        probAnalysis.weightedMeanApriori.accuracy = 1 - probAnalysis.weightedMeanApriori.misclassification;

        % Calculate AUC for each class
        probAnalysis.weightedMeanApriori.auc = zeros(1,size(featuresTestingLabelAsMatrix,2));
        for cntClass = 1 : size(featuresTestingLabelAsMatrix,2)
            [probAnalysis.weightedMeanApriori.auc(cntClass),~,~] = fastAUC(logical(featuresTestingLabelAsMatrix(:,cntClass)),probAnalysis.weightedMeanApriori.Probabilities(:,cntClass),ctrl.plotROC);
        end

        %% Series probability
        probAnalysis.seriesPredictor = struct();

        if ctrl.calcSeriesPrediction
            % Calc probability
            [probAnalysis.seriesPredictor.Probabilities, probAnalysis.seriesPredictor.Sequence] = calcSeriesProbabilityForEachMeasurement(featuresTesting, featuresTraining, trainedClassifier);

            % Apply weighted mean
            [probAnalysis.seriesPredictor.Probabilities, probAnalysis.seriesPredictor.Sequence] = calcMeanProbabilityForEachMeasurement(featuresTesting, 'aprioriProbability', probAnalysis.seriesPredictor.Probabilities);

            % Calculate m-value (quasi multiclass AUC value)
            probAnalysis.seriesPredictor.mValue = multiClassAUC(probAnalysis.seriesPredictor.Probabilities,trueClassSequence);

            % Calculate accuracy
            [probAnalysis.seriesPredictor.misclassification,~,~,~] = confusion(featuresTestingLabelAsMatrix', probAnalysis.seriesPredictor.Probabilities');
            probAnalysis.seriesPredictor.accuracy = 1 - probAnalysis.seriesPredictor.misclassification;

            % Calculate AUC for each class
            probAnalysis.seriesPredictor.auc = zeros(1,size(featuresTestingLabelAsMatrix,2));
            for cntClass = 1 : size(featuresTestingLabelAsMatrix,2)
                [probAnalysis.seriesPredictor.auc(cntClass),~,~] = fastAUC(logical(featuresTestingLabelAsMatrix(:,cntClass)),probAnalysis.seriesPredictor.Probabilities(:,cntClass),ctrl.plotROC);
            end

        else
            probAnalysis.seriesPredictor.mValue = NaN;
            probAnalysis.seriesPredictor.accuracy = NaN;
            probAnalysis.seriesPredictor.auc = [NaN NaN NaN NaN];
        end    


        %% Hidden Markov Model
        probAnalysis.hiddenMarkov = struct();

        if ctrl.calcHiddenMarkovModel
            % Train model
            if exist('featuresTrainingData')
                [mu, sigma, TRANS] = trainHiddenMarkovModel(featuresTrainingLabelAsMatrix, 'features', featuresTrainingData, 'trainedClassifier', tmpTrainedClassifier{cntCV});
            else
                [mu, sigma, TRANS] = trainHiddenMarkovModel(featuresTrainingLabelAsMatrix, 'aprioriProbabilityMatrix', featuresTrainingPosterior);
            end
            % Calc probability
            [probAnalysis.hiddenMarkov.Probabilities, probAnalysis.hiddenMarkov.Sequence] = calcMarkovProbabilityForEachMeasurement(mu, sigma, TRANS, featuresTestingPosterior, featuresTestingBatch);

            % Apply weighted mean
%             [probAnalysis.hiddenMarkov.Probabilities, probAnalysis.hiddenMarkov.Sequence] = calcMeanProbabilityForEachMeasurement(probAnalysis.hiddenMarkov.Probabilities, featuresTestingBatch);

            % Calculate m-value (quasi multiclass AUC value)
            probAnalysis.hiddenMarkov.mValue = multiClassAUC(probAnalysis.hiddenMarkov.Probabilities,trueClassSequence);

            % Calculate accuracy
            [probAnalysis.hiddenMarkov.misclassification,~,~,~] = confusion(featuresTestingLabelAsMatrix', probAnalysis.hiddenMarkov.Probabilities');
            probAnalysis.hiddenMarkov.accuracy = 1 - probAnalysis.hiddenMarkov.misclassification;

            % Calculate AUC for each class
            probAnalysis.hiddenMarkov.auc = zeros(1,size(featuresTestingLabelAsMatrix,2));
            for cntClass = 1 : size(featuresTestingLabelAsMatrix,2)
                [probAnalysis.hiddenMarkov.auc(cntClass),~,~] = fastAUC(logical(featuresTestingLabelAsMatrix(:,cntClass)),probAnalysis.hiddenMarkov.Probabilities(:,cntClass),ctrl.plotROC);
            end

        else
            probAnalysis.hiddenMarkov.mValue = NaN;
            probAnalysis.hiddenMarkov.accuracy = NaN;
            probAnalysis.hiddenMarkov.auc = [NaN NaN NaN NaN];
        end

        probAnalysisOut{cntCV} = probAnalysis;
        
    end

    %% Merge result of probability analysis (needed for parfor)
    probAnalysis = probAnalysisOut{1};
    for cntCV = 2 : length(probAnalysisOut)
        probAnalysis = mergeStructs(probAnalysis, probAnalysisOut{cntCV});
    end
    
    % Reconvert Probabilities and Sequence to cross-validation style
    fieldsProb = fields(probAnalysis);
    for cntFields = 1 : length(fieldsProb)
        if isfield(probAnalysis.(fieldsProb{cntFields}),'Probabilities')
            tmpProbabilities = probAnalysis.(fieldsProb{cntFields}).Probabilities;
            tmpSequence = probAnalysis.(fieldsProb{cntFields}).Sequence;
            probAnalysis.(fieldsProb{cntFields}).Probabilities = cell(length(probAnalysisOut),1);
            probAnalysis.(fieldsProb{cntFields}).Sequence = cell(length(probAnalysisOut),1);
            nObs = size(tmpProbabilities,1)/length(probAnalysisOut);
            for cntCV = 1 : length(probAnalysisOut)
                probAnalysis.(fieldsProb{cntFields}).Probabilities{cntCV} = tmpProbabilities((cntCV-1)*nObs+1:(cntCV*nObs),:);
                probAnalysis.(fieldsProb{cntFields}).Sequence{cntCV} = tmpSequence((cntCV-1)*nObs+1:(cntCV*nObs),:);
            end
        end
    end
        
    fieldsProbAnalysis = fields(probAnalysis);
    for cntField = 1 : length(fieldsProbAnalysis)
        tmpData = probAnalysis.(fieldsProbAnalysis{cntField}).mValue;
        probAnalysis.(fieldsProbAnalysis{cntField}).mValue = struct('raw', tmpData, 'mean', mean(tmpData), 'std', std(tmpData));
        
        tmpData = probAnalysis.(fieldsProbAnalysis{cntField}).accuracy;
        probAnalysis.(fieldsProbAnalysis{cntField}).accuracy = struct('raw', tmpData, 'mean', mean(tmpData), 'std', std(tmpData));
    end
    
    %% Plot
    figure;
    hold on
    bar([probAnalysis.apriori.mValue.mean, probAnalysis.meanApriori.mValue.mean, probAnalysis.movMeanApriori.mValue.mean, probAnalysis.weightedMeanApriori.mValue.mean, probAnalysis.seriesPredictor.mValue.mean, probAnalysis.hiddenMarkov.mValue.mean;...
        probAnalysis.apriori.accuracy.mean, probAnalysis.meanApriori.accuracy.mean, probAnalysis.movMeanApriori.accuracy.mean, probAnalysis.weightedMeanApriori.accuracy.mean, probAnalysis.seriesPredictor.accuracy.mean, probAnalysis.hiddenMarkov.accuracy.mean])
    set(gca,'XTick',[1 2]);
    set(gca,'XTickLabel',{'mValue';'Accuracy'});
    legend({'apriori';'mean apriori';'movmean apriori';'weightedmean apriori';'Series';'Markov'})
    legend show
    
    %% Plot Confusion Matrix
    if ctrl.plotConfusionMatrix
    
        figure;
        plotconfusion(categorical(trueClassSequence),categorical(probAnalysis.apriori.Sequence),['a Prior ']);
        if isstruct(featuresTesting)
            set(gca,'XTickLabel',[featuresTesting.uniqueClasses;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[featuresTesting.uniqueClasses;' ']);
        end
        
        if ~isempty(featuresTraining)
            if ctrl.calcHiddenMarkovModel
                figure;
                plotconfusion(categorical(trueClassSequence),categorical(probAnalysis.hiddenMarkov.Sequence),['Posterior Markov ']);
                if isstruct(featuresTesting)
                    set(gca,'XTickLabel',[featuresTesting.uniqueClasses;' ']);
                    set(gca,'XTickLabelRotation',60);
                    set(gca,'YTickLabel',[featuresTesting.uniqueClasses;' ']);
                end
            end
        end

        figure;
        plotconfusion(categorical(trueClassSequence),categorical(probAnalysis.weightedMeanApriori.Sequence),['Posterior weightedmean ']);
        if isstruct(featuresTesting)
            set(gca,'XTickLabel',[featuresTesting.uniqueClasses;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[featuresTesting.uniqueClasses;' ']);
        end

        figure;
        plotconfusion(categorical(trueClassSequence),categorical(probAnalysis.movMeanApriori.Sequence),['Posterior movmean ']);
        if isstruct(featuresTesting)
            set(gca,'XTickLabel',[featuresTesting.uniqueClasses;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[featuresTesting.uniqueClasses;' ']);
        end

        figure;
        plotconfusion(categorical(trueClassSequence),categorical(probAnalysis.meanApriori.Sequence),['Posterior mean ']);
        if isstruct(featuresTesting)
            set(gca,'XTickLabel',[featuresTesting.uniqueClasses;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[featuresTesting.uniqueClasses;' ']);
        end
        
        if ctrl.calcSeriesPrediction
            figure;
            plotconfusion(categorical(trueClassSequence),categorical(probAnalysis.seriesPredictor.Sequence),['Posterior series ']);
            if isstruct(featuresTesting)
                set(gca,'XTickLabel',[featuresTesting.uniqueClasses;' ']);
                set(gca,'XTickLabelRotation',60);
                set(gca,'YTickLabel',[featuresTesting.uniqueClasses;' ']);
            end
        end

    end
    
    %% Plot measurement analysis
    if ctrl.plotMeasurementAnalysis
        plotCV = 1;
        figure;
        ax1 = subplot(2,1,1);
        plot(max(probAnalysis.apriori.Probabilities{plotCV},[],2),'Color',[1,0.67,0.67],'DisplayName',['max-apriori ']);
        hold on;
        plot(max(probAnalysis.apriori.Probabilities{plotCV}.*featuresTesting.labelAsMatrix,[],2),'Color',[0.8 0.8 0.8],'DisplayName','trueClass-apriori');
        plot(max(probAnalysis.meanApriori.Probabilities{plotCV},[],2),'LineWidth',1.5,'DisplayName',['mean ']);
        if ~isnan(probAnalysis.seriesPredictor.accuracy.mean)
            plot(max(probAnalysis.seriesPredictor.Probabilities{plotCV},[],2),'LineWidth',1.5,'DisplayName',['series ']);
        end
        if ~isnan(probAnalysis.hiddenMarkov.accuracy.mean)
            plot(max(probAnalysis.hiddenMarkov.Probabilities{plotCV},[],2),'LineWidth',1.5,'DisplayName',['Markov ']);
        end
        legend show

        ax2 = subplot(2,1,2);
        plot(trueClassSequence,'Color','k','LineWidth',4,'DisplayName','True');
        hold on;
        plot(probAnalysis.apriori.Sequence{plotCV},'Color',[0.8 0.8 0.8],'DisplayName','apriori');
        plot(probAnalysis.meanApriori.Sequence{plotCV},'LineWidth',1.5,'DisplayName','mean');
        if ~isnan(probAnalysis.seriesPredictor.accuracy.mean)
            plot(probAnalysis.seriesPredictor.Sequence{plotCV},'LineWidth',1.5,'DisplayName','series')
        end
        if ~isnan(probAnalysis.hiddenMarkov.accuracy.mean)
            plot(probAnalysis.hiddenMarkov.Sequence{plotCV},'LineWidth',1.5,'DisplayName','Markov')
        end
        legend show
        linkaxes([ax1, ax2], 'x')
    end
    
end

