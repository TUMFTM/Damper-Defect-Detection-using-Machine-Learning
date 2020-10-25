function [accuracyOut, numFeaturesVector] = predictForEachFeatureSubset(featuresTesting, classifierCellArray, varargin)
%TRAINSVMFOREACHFEATURESUBSET Summary of this function goes here
%   Predict for each number of features and plot accuracy for test data
%   The selected features need to be defined within the classifier

    if find(strcmp(varargin,'plotOutput'))
        plotOutput = varargin{find(strcmp(varargin,'plotOutput'))+1};
    else
        plotOutput = 1;
    end

    numFeaturesVector = 1 : length(classifierCellArray);
    accuracyUsingSelectedFeatures = cell(length(classifierCellArray),1);
    
    % Copy for faster parfor run
    data = featuresTesting.data; 
    labelAsMatrix = featuresTesting.labelAsMatrix;
    Label = featuresTesting.Label;
    
    parfor idxNumFeatures = 1 : length(classifierCellArray)
        
        if ~isempty(classifierCellArray{idxNumFeatures})
            % predict and get accuracy
            predictedClass = predictClassifier(classifierCellArray{idxNumFeatures}, data);
            if length(predictedClass) == length(classifierCellArray{idxNumFeatures})
                misclassification = zeros(size(predictedClass));
                for cntCV = 1 : length(predictedClass)
                    misclassification(cntCV) = sum(~strcmp(Label,predictedClass{cntCV}))/length(Label);
                end
            else
                misclassification = sum(~strcmp(Label,predictedClass))/length(Label);
            end
            accuracyUsingSelectedFeatures{idxNumFeatures,1} = 100*(1-misclassification);
        else
            accuracyUsingSelectedFeatures{idxNumFeatures,1} = NaN;
        end
        
    end
    
    accuracyOut.mean = cellfun(@mean,accuracyUsingSelectedFeatures);
    accuracyOut.std = cellfun(@std,accuracyUsingSelectedFeatures);
    
    if plotOutput
        % Plot accuracy
        figure;
        errorbar(numFeaturesVector, accuracyOut.mean, accuracyOut.std);
        xlabel('Number of Features selected');
        ylabel('Accuracy');
    end
end

