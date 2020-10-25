function plot_LDS_Analysis(testData)
%PLOT_LDS_ANALYSIS Summary of this function goes here
%   Detailed explanation goes here
% Plot visualization of post processing

    plotDataset = 'test';

    tmpClassIdx = [1 : size(testData.(plotDataset).features.labelAsMatrix,2)]';
    trueClassSequence = testData.(plotDataset).features.labelAsMatrix * tmpClassIdx;
    
    novScores = table;
    if isfield(testData.(plotDataset).features.Prop, 'ID')
        novScores.batch = testData.(plotDataset).features.Prop.ID;
    else
        novScores.batch = testData.(plotDataset).features.Prop.batch;
    end
    novScores.obsID = testData.(plotDataset).features.Prop.observationID;
    novScores.labelID = trueClassSequence;
    novScores.pre = testData.(plotDataset).features.Prop.Posterior{1};
    novScores.post = testData.(plotDataset).probabilityAnalysis.LDS.Probabilities{1};
    novScores = sortrows(novScores);
    
    figure;
    ax(1) = subplot(2,1,1);
    h(1)=plot(novScores.pre,'Color',[204 204 204]/255,'LineWidth', 1,'DisplayName','Apriori');
    hold on
    newMeas = [0; diff(novScores.batch)];
    idxNewMeas = find(newMeas);
    for cntIdx = 1 : length(idxNewMeas)
        htmp = plot([idxNewMeas(cntIdx) idxNewMeas(cntIdx)],[0 max(novScores.pre)],'-.','Color',[0 0 0]/255,'DisplayName','New Measurement');
        if cntIdx == 1
            h(end+1) = htmp;
        end
    end
    uniqueBatches = unique(novScores.batch);
    runningNumObs = 1 : length(novScores.obsID);
    for cntBatch = 1 : length(uniqueBatches)
        idxBatch = novScores.batch == uniqueBatches(cntBatch);
        htmp=plot(runningNumObs(idxBatch),novScores.post(idxBatch),'k-','LineWidth', 1.5,'DisplayName','Posteriori');
        if cntBatch == 1
            h(end+1) = htmp;
        end
    end

    legend(h,'Interpreter','None')
    ylabel('Novelty Score')
    xlabel('Observation Index')
    title('Post-Processing by Kalman Filter')

    ax(2) = subplot(2,1,2);
    h(end+1) = plot(novScores.labelID, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True Class');
    hold on
    newMeas = [0; diff(novScores.batch)];
    idxNewMeas = find(newMeas);

    set(gca,'YTick', tmpClassIdx);
    set(gca,'YTickLabel', testData.(plotDataset).features.uniqueClasses);
    ylim([0.9 size(testData.(plotDataset).features.labelAsMatrix,2)+0.1]);
    ylabel('Class');
    xlabel('Observation Index')
    
    linkaxes(ax,'x');
end

