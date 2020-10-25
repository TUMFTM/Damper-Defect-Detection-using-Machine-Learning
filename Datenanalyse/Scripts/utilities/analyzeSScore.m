function SScore = analyzeSScore(features)
%ANALYZEFISHERSCORE Summary of this function goes here
%   Detailed explanation goes here
    SScore = struct();
    SScore.Score = table;
    SScore.Score.Name = features.data.Properties.VariableNames';
    SScore.Score.SScore = calcFisherScore(features.data, features.labelAsMatrix);

    SScore.sortedByScore = sortrows(SScore.Score,'SScore','descend');

    figure;
    plot(SScore.sortedByScore.SScore,'DisplayName','S-Score');
    legend show
    xlabel('Feature Index');
    ylabel('S-Score');
end

