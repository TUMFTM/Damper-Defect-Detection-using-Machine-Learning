function FisherScore = analyzeFisherScore(features)
%ANALYZEFISHERSCORE Summary of this function goes here
%   Detailed explanation goes here
    FisherScore = struct();
    FisherScore.Score = table;
    if istable(features.data)
        FisherScore.Score.Name = features.data.Properties.VariableNames';
    else
        FisherScore.Score.Name = features.featureNames';
    end
    FisherScore.Score.FisherScore = calcFisherScore(features.data, features.labelAsMatrix);

    FisherScore.Score.FeatBlockName = extractBefore(FisherScore.Score.Name, '_');
    
    FisherScore.sortedByScore = sortrows(FisherScore.Score,'FisherScore','descend');

    figure;
    boxplot(FisherScore.Score.FisherScore, FisherScore.Score.FeatBlockName);
    
    figure;
    plot(FisherScore.sortedByScore.FisherScore,'DisplayName','Fisher Score');
    legend show
    xlabel('Feature Index');
    ylabel('Fisher Score');
    title('Feature Importance using Fisher Score');
    
    figure
    plot(FisherScore.Score.FisherScore,'DisplayName','Score');
    legend show
    xlabel('Original Feature Index');
    ylabel('Fisher Score');
    title('Feature Importance using Fisher Score');
end

