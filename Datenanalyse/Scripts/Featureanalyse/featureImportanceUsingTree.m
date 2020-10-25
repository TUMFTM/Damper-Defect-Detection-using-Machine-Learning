function [TreeImportance] = featureImportanceUsingTree(varargin)
%FEATUREIMPORTANCEUSINGTREE Estimates feature importance using a Random Forest
%   Detailed explanation goes here

    %% Read input arguments
    if find(strcmp(varargin,'trainedClassifier'))
        trainedClassifier = varargin{find(strcmp(varargin,'trainedClassifier'))+1};
    else
        if find(strcmp(varargin,'featuresTraining'))
            featuresTraining = varargin{find(strcmp(varargin,'featuresTraining'))+1};
            classifierOptsTree = setOptsClassifier('Tree');
            trainedClassifier = trainClassifier(featuresTraining, classifierOptsTree);
        else
            fprintf('Unexpected input to function featureImportanceUsingTree\n');
        end
    end

    %% Perform feature importance estimation
    TreeImportance = struct();
    TreeImportance.trainedClassifier = trainedClassifier;
    TreeImportance.Score = table;
    TreeImportance.Score.Name = trainedClassifier.ExpandedPredictorNames';
    
    [imp, ~] = predictorImportance(trainedClassifier);
    TreeImportance.Score.Score = imp';

    TreeImportance.sortedByScore = sortrows(TreeImportance.Score,'Score','descend');

    figure;
    subplot(2,1,1);
    plot(TreeImportance.sortedByScore.Score,'DisplayName','Score');
    legend show
    xlabel('Feature Index Sorted');
    ylabel('Score');
    title('Feature Importance using a Random Forest');
    
    subplot(2,1,2);
    plot(TreeImportance.Score.Score,'DisplayName','Score');
    legend show
    xlabel('Original Feature Index');
    ylabel('Score');

end

