function analyzeHistogram(VarToBeAnalyzed, ScoresAsMatrix, labelAsMatrix)
%ANALYZEHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here

% Inputs:
% VarToBeAnalyzed = features.Prop.meanAx
% ScoresAsMatrix = features.Prop.Posterior
% labelAsMatrix = features.labelAsMatrix

% Example call: analyzeHistogram(testDD2MassCNN.features.Prop.meanAx, testDD2MassCNN.features.Prop.Posterior, testDD2MassCNN.features.labelAsMatrix);
    
    scoreThreshold = 0.3;

    figure;
    histogram(VarToBeAnalyzed(sum(ScoresAsMatrix.*labelAsMatrix,2)>scoreThreshold),'FaceColor','g','Normalization','probability');
    hold on;
    histogram(VarToBeAnalyzed(~(sum(ScoresAsMatrix.*labelAsMatrix,2)>scoreThreshold)),'FaceColor','r','Normalization','probability');
    legend('correct Pred.','wrong Pred.')
end

