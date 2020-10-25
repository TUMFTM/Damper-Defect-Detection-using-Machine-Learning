function [posteriorClassSequence, MarkovPosteriorProbabilityMatrix] = evaluateHiddenMarkovModel(mu, sigma, TRANS, varargin)
%TRAINHIDDENMARKOVMODEL Summary of this function goes here
%   Detailed explanation goes here

    ctrl.plotConfusionMatrix = 1;

    %% Einlesen der übergebenen Größen
    if find(strcmp(varargin,'features'))
        features = varargin{find(strcmp(varargin,'features'))+1};
    else
        features = 0;
    end
    
    if find(strcmp(varargin,'trueClassSequence'))
        trueClassSequence = varargin{find(strcmp(varargin,'trueClassSequence'))+1};
        if size(trueClassSequence,2) > 1
            n_classes = size(trueClassSequence,2);
            tmpClassIdx = [1 : n_classes]';
            trueClassSequence = trueClassSequence * tmpClassIdx;
        end
    else
        trueClassSequence = 0;
    end
    
    if find(strcmp(varargin,'aprioriProbabilityMatrix'))
        aprioriPredictedProbabilityMatrix = varargin{find(strcmp(varargin,'aprioriProbabilityMatrix'))+1};
    else
        aprioriPredictedProbabilityMatrix = 0;
    end

    % Generate apriori Probability Matrix
    if isstruct(features)
        if isfield(features.Prop, 'Posterior')
            aprioriPredictedProbabilityMatrix = features.Prop.Posterior;
        else
            [~, aprioriPredictedProbabilityMatrix] = predictClassifier(trainedClassifier, table2array(features.data));
        end
    end
    
    % Generate number of classes
    n_classes = size(aprioriPredictedProbabilityMatrix,2);
    
    % Generate TrueClass sequence
    if isstruct(features)
        if isfield(features, 'labelAsMatrix')
            tmpClassIdx = [1 : length(features.uniqueClasses)]';
            trueClassSequence = features.labelAsMatrix * tmpClassIdx;
        end
    end
    
    [~,aprioriPredictedClassSequence] = max(aprioriPredictedProbabilityMatrix,[],2);
    
    posteriorClassSequence = [];
    posteriorProbabilityMatrix1 = [];
    posteriorProbabilityMatrix2 = [];
    weightedMeanAprioriProbability = [];
    meanAprioriProbability = [];
    movmeanAprioriProbability = [];
    
    for cntBatch = 1 : max(features.Prop.batch)
        idxMeasurement = logical(features.Prop.batch == cntBatch);
        
        tmpPosteriorClassSequence = hmmviterbi(aprioriPredictedClassSequence(idxMeasurement,:),TRANS, EMIS)';
        posteriorClassSequence = [posteriorClassSequence; tmpPosteriorClassSequence];
        
        tmpPosteriorProbabilityMatrix2 = hmmdecode(aprioriPredictedClassSequence(idxMeasurement,:)',TRANS, EMIS)';
        posteriorProbabilityMatrix2 = [posteriorProbabilityMatrix2; tmpPosteriorProbabilityMatrix2];
        
        % Calculate 'weight' for weighted mean apriori probability
        % 'weight' has same length as aprioirPredictedProbabilityMatrix
        % sum(weight) = 1
        maxAprioirProbability = max(aprioriPredictedProbabilityMatrix,[],2);
        idxMaxProbApriorProb = bsxfun(@eq, aprioriPredictedProbabilityMatrix, maxAprioirProbability);
        tmp = aprioriPredictedProbabilityMatrix;
        tmp(idxMaxProbApriorProb) = 0;
        maxPorb2 = max(tmp,[],2);
        weight = zeros(size(aprioriPredictedProbabilityMatrix,1),1);
        weight(idxMeasurement) = (maxAprioirProbability(idxMeasurement) - maxPorb2(idxMeasurement)) ./ sum(maxAprioirProbability(idxMeasurement) - maxPorb2(idxMeasurement));
        
        % 'tmpMeanAprioriProbability' has the weighted mean of the
        % aprioirProbability of a whole measurement for each observation
        tmpWeightedMeanAprioriProbability = ones(sum(idxMeasurement),1)*(weight(idxMeasurement)' * aprioriPredictedProbabilityMatrix(idxMeasurement,:));
        weightedMeanAprioriProbability = [weightedMeanAprioriProbability; tmpWeightedMeanAprioriProbability];
        
        % Mean of aprioir Probabilities
        tmpMeanAprioriProbability = ones(sum(idxMeasurement),1).*mean(aprioriPredictedProbabilityMatrix(idxMeasurement,:));
        meanAprioriProbability = [meanAprioriProbability; tmpMeanAprioriProbability];
        
        % Mov Mean of apriori probabilities
        tmpMovMeanAprioriProbability = movmean(aprioriPredictedProbabilityMatrix(idxMeasurement,:),[100000,0]);
        movmeanAprioriProbability = [movmeanAprioriProbability; tmpMovMeanAprioriProbability];
    end
    
    MarkovPosteriorProbabilityMatrix = posteriorProbabilityMatrix2;
    if size(posteriorProbabilityMatrix2,2)>n_classes
        MarkovPosteriorProbabilityMatrix = MarkovPosteriorProbabilityMatrix(:,2:end);
    end
    
    [~,MarkovPredictedClassSequence] = max(MarkovPosteriorProbabilityMatrix,[],2);
    [~,weightedMeanPredictedClassSequence] = max(weightedMeanAprioriProbability,[],2);
    [~,movmeanPredictedClassSequence] = max(movmeanAprioriProbability,[],2);
    [~,meanAprioriPredictedClassSequence] = max(meanAprioriProbability,[],2);
    
    % Calculate m-value according to Hand DJ, Till RJ, "A Simple Generalization of the Area Under the ROC Curve
            %                       for Multiple Class Classification Problems, Machine Learning, 45, 171-186,
            %                       2001.
    mAprioriPredictedProbability = multiClassAUC(aprioriPredictedProbabilityMatrix,trueClassSequence);
    mWeightedMeanAprioriProbability = multiClassAUC(weightedMeanAprioriProbability,trueClassSequence);
    mMovmeanMeanAprioriProbability = multiClassAUC(movmeanAprioriProbability,trueClassSequence);
    mMarkovPosteriorProbability = multiClassAUC(MarkovPosteriorProbabilityMatrix,trueClassSequence);
    mMeanAprioriProbability = multiClassAUC(meanAprioriProbability,trueClassSequence);
    
    figure;
    ax1 = subplot(2,1,1);
    hold on;
    plot(max(aprioriPredictedProbabilityMatrix,[],2),'Color',[1,0.67,0.67],'DisplayName',['max-apriori ', num2str(mAprioriPredictedProbability)]);
    plot(max(aprioriPredictedProbabilityMatrix.*features.labelAsMatrix,[],2),'Color',[0.8 0.8 0.8],'DisplayName','trueClass-apriori');
    plot(max(movmeanAprioriProbability,[],2),'LineWidth',1.5,'DisplayName',['movmean ', num2str(mMovmeanMeanAprioriProbability)]);
    plot(max(meanAprioriProbability,[],2),'LineWidth',1.5,'DisplayName',['mean ', num2str(mMeanAprioriProbability)]);
    plot(max(weightedMeanAprioriProbability,[],2),'LineWidth',1.5,'DisplayName',['weightedmean ', num2str(mWeightedMeanAprioriProbability)]);
    plot(max(MarkovPosteriorProbabilityMatrix,[],2),'LineWidth',1.5,'DisplayName',['Markov ', num2str(mMarkovPosteriorProbability)]);
    legend show
    
    ax2 = subplot(2,1,2);
    hold on;
    plot(trueClassSequence,'Color','k','LineWidth',2,'DisplayName','True');
    plot(aprioriPredictedClassSequence,'Color',[0.8 0.8 0.8],'DisplayName','apriori');
    plot(movmeanPredictedClassSequence,'LineWidth',1.5,'DisplayName','movmean');
    plot(meanAprioriPredictedClassSequence,'LineWidth',1.5,'DisplayName','mean');
    plot(weightedMeanPredictedClassSequence,'LineWidth',1.5,'DisplayName','weightedmean')
    plot(MarkovPredictedClassSequence,'LineWidth',1.5,'DisplayName','Markov')
    legend show
    linkaxes([ax1, ax2], 'x')
    
    %% Plot Confusion Matrix
    if length(TRANS) > n_classes
        posteriorClassSequence = posteriorClassSequence - 1;
    end
    
    if ctrl.plotConfusionMatrix
    
        figure;
        plotconfusion(categorical(trueClassSequence),categorical(aprioriPredictedClassSequence),['a Prior ', num2str(mAprioriPredictedProbability)]);
        if isstruct(features)
            set(gca,'XTickLabel',[features.uniqueClasses;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[features.uniqueClasses;' ']);
        end

        figure;
        plotconfusion(categorical(trueClassSequence),categorical(MarkovPredictedClassSequence),['Posterior Markov ', num2str(mMarkovPosteriorProbability)]);
        if isstruct(features)
            set(gca,'XTickLabel',[features.uniqueClasses;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[features.uniqueClasses;' ']);
        end

        figure;
        plotconfusion(categorical(trueClassSequence),categorical(weightedMeanPredictedClassSequence),['Posterior weightedmean ', num2str(mWeightedMeanAprioriProbability)]);
        if isstruct(features)
            set(gca,'XTickLabel',[features.uniqueClasses;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[features.uniqueClasses;' ']);
        end

        figure;
        plotconfusion(categorical(trueClassSequence),categorical(movmeanPredictedClassSequence),['Posterior movmean ', num2str(mMovmeanMeanAprioriProbability)]);
        if isstruct(features)
            set(gca,'XTickLabel',[features.uniqueClasses;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[features.uniqueClasses;' ']);
        end

        figure;
        plotconfusion(categorical(trueClassSequence),categorical(meanAprioriPredictedClassSequence),['Posterior mean ', num2str(mMeanAprioriProbability)]);
        if isstruct(features)
            set(gca,'XTickLabel',[features.uniqueClasses;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[features.uniqueClasses;' ']);
        end

    end

end

