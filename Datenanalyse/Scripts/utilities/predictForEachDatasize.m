function [accuracyOut, numFeaturesVector] = predictForEachDatasize(dataTesting, datasizeAnalysis, varargin)
%TRAINSVMFOREACHFEATURESUBSET Summary of this function goes here
%   Predict for each number of features and plot accuracy for test data
%   The selected features need to be defined within the classifier

    if find(strcmp(varargin,'plotOutput'))
        plotOutput = varargin{find(strcmp(varargin,'plotOutput'))+1};
    else
        plotOutput = 1;
    end

    
    
    % Copy for faster parfor run
    featureExtractionHandle = datasizeAnalysis.featureExtractionHandle;
    classifierCellArray = datasizeAnalysis.trainedClassifier;
%     data = dataTesting.data; 
%     labelAsMatrix = dataTesting.labelAsMatrix;
%     Label = dataTesting.Label;
    
    numFeaturesVector = 1 : length(classifierCellArray);
    accuracyUsingSelectedFeatures = cell(length(classifierCellArray),1);
    
    for idxNumFeatures = 1 : length(classifierCellArray)
        
        features = generateFeatureStruct(dataTesting, featureExtractionHandle{idxNumFeatures,1}, 'uniqueClasses', classifierCellArray{idxNumFeatures,1}{1,1}.ClassNames);
        
        if ~isempty(classifierCellArray{idxNumFeatures,1})
            % predict and get accuracy
            predictedClass = predictClassifier(classifierCellArray{idxNumFeatures,1}, features.data);
            if length(predictedClass) == length(classifierCellArray{idxNumFeatures,1})
                misclassification = zeros(size(predictedClass));
                for cntCV = 1 : length(predictedClass)
                    misclassification(cntCV) = sum(~strcmp(features.Label,predictedClass{cntCV}))/length(features.Label);
                end
            else
                misclassification = sum(~strcmp(features.Label,predictedClass))/length(features.Label);
            end
            
%             [~, predictedProbability] = predictClassifier(classifierCellArray{idxNumFeatures}, data);
%             if iscell(predictedProbability)
%                 misclassification = zeros(size(predictedProbability));
%                 for cntCV = 1 : length(predictedProbability)
%                     [misclassification(cntCV),~,~,~] = confusion(labelAsMatrix', predictedProbability{cntCV}');
%                 end
%             else
%                 [misclassification,~,~,~] = confusion(labelAsMatrix', predictedProbability');
%             end
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

