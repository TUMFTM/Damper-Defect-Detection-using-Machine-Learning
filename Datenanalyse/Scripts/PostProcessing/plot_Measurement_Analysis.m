function plot_Measurement_Analysis(probAnalysis, featuresTesting)
%PLOT_MEASUREMENT_ANALYSIS Summary of this function goes here
%   Detailed explanation goes here
    plotCV = 1;
    
    tmpClassIdx = [1 : size(featuresTesting.labelAsMatrix,2)]';
    trueClassSequence = featuresTesting.labelAsMatrix * tmpClassIdx;
    
    scores = table;
    if isfield(featuresTesting.Prop, 'ID')
        scores.batch = featuresTesting.Prop.ID;
    else
        scores.batch = featuresTesting.Prop.batch;
    end
    scores.obsID = featuresTesting.Prop.observationID;
    scores.labelID = featuresTesting.Prop.LabelID;
    scores.label = featuresTesting.Prop.labelIsolation;
    
    scores.aprioriPredicted = max(probAnalysis.apriori.Probabilities{plotCV},[],2);
    scores.aprioriTrueClass = max(probAnalysis.apriori.Probabilities{plotCV}.*featuresTesting.labelAsMatrix,[],2);
    scores.aprioriMean = max(probAnalysis.meanApriori.Probabilities{plotCV},[],2);
    scores.posterioriHMM = max(probAnalysis.hiddenMarkov.Probabilities{plotCV},[],2);
    
    scores.trueClassSequence = featuresTesting.labelAsMatrix * tmpClassIdx;
    scores.aprioriSequence = probAnalysis.apriori.Sequence{plotCV};
    scores.aprioriMeanSequence = probAnalysis.meanApriori.Sequence{plotCV};
    scores.posterioriHMMSequence = probAnalysis.hiddenMarkov.Sequence{plotCV};

    scores = sortrows(scores);
    
    scores.aprioriPredictedCorrect = scores.aprioriPredicted;
    scores.aprioriPredictedWrong = scores.aprioriPredicted;
    scores.aprioriPredictedWrong(scores.aprioriPredicted==scores.aprioriTrueClass) = NaN;

    figure;
    ax1 = subplot(2,1,1);
    hold on
    plot(100*scores.aprioriPredicted,'Color',[204 204 204]/255,'LineWidth',0.5,'DisplayName','apriori predicted class');
    plot(100*scores.aprioriMean,'-.k','LineWidth',1.5,'DisplayName',['apriori mean']);
    plot(100*scores.posterioriHMM,'-k','LineWidth',1.5,'DisplayName','HMM');
    legend show
    title('Estimated Probabilities');
    xlabel('Observation Index');
    ylabel('Probability in %');
    
    ax2 = subplot(2,1,2);
    plot(scores.aprioriSequence,'Color',[204 204 204]/255,'DisplayName','apriori predicted class');
    hold on;
    plot(scores.trueClassSequence,'Color','k','LineWidth',4,'LineStyle','--','DisplayName','true class');
    plot(scores.aprioriMeanSequence,'-.k','LineWidth',1.5,'DisplayName','apriori mean');
    plot(scores.posterioriHMMSequence,'k','LineWidth',1.5,'DisplayName','HMM')
    legend show
    linkaxes([ax1, ax2], 'x')
    set(gca, 'YTick', [1:length(featuresTesting.uniqueClasses)])
    set(gca, 'YTickLabels',featuresTesting.uniqueClasses);
    title('Estimated Class')
	xlabel('Observation Index')
    ylabel('Class Index');

end

