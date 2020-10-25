function RFE = performRFEusingWeights(featuresTraining, varargin)
%PERFORMRFE Performs a recursive feature elimination for feature selection

% features are selected based on the weights
% of a linear SVM according to I. Guyon, J. Weston, S. Barnhill, and V. Vapnik, �Gene Selection for Cancer Classification using Support Vector Machines,� Machine Learning, vol. 46, no. 1/3, pp. 389�422, 2002.
% This method does not allow to use testing data for feature selection
% requires n trainings of a linear SVM (n = number of features)


    %% Read additional input arguments
    if find(strcmp(varargin,'classifierType'))
        classifierType = varargin{find(strcmp(varargin,'classifierType'))+1};
    else
        classifierType = 'SVM';
    end

    %% Acutal calculation
    classifierOpts = setOptsClassifier(classifierType);
    classifierOpts.optimizeHyperparameters = '';    % disable classifier hyperparameter optimization
    classifierOpts.validation = 0.2;    % disable cross-validation
    
    if ~isfield(featuresTraining, 'featureNames')
        featuresTraining.featureNames = featuresTraining.data.Properties.VariableNames';
    end
    tmpRFEFeatureList = featuresTraining.featureNames';

    numFeatures = length(featuresTraining.featureNames);

    RFE = struct();
    RFE.sortedByRank = struct();
    RFE.sortedByRank.Name = [];
    RFE.featureSubsetNames = cell(numFeatures,1);
 
    for cntNumFeatures = numFeatures : -1 : 2

        trainedClassifier = trainClassifier(featuresTraining, classifierOpts, 'TextOutput', 1, 'selectedFeatures', tmpRFEFeatureList, 'FitPosterior', false);
        RFE.trainedClassifier{cntNumFeatures,1} = trainedClassifier;
        RFE.featureSubsetNames{cntNumFeatures,1} = RFE.trainedClassifier{cntNumFeatures,1}.ExpandedPredictorNames';
        
        avgAbsWeight = zeros(size(trainedClassifier.BinaryLearners{1,1}.Beta));
        for cntLearners = 1 : length(trainedClassifier.BinaryLearners)
            avgAbsWeight = avgAbsWeight + abs(trainedClassifier.BinaryLearners{cntLearners,1}.Beta);
        end
        avgAbsWeight = avgAbsWeight ./ length(trainedClassifier.BinaryLearners);
        [~,idxMinAvgAbsWeight] = min(avgAbsWeight);

        RFE.sortedByRank.Name{cntNumFeatures,1} = tmpRFEFeatureList{idxMinAvgAbsWeight};
        
        tmpRFEFeatureList(idxMinAvgAbsWeight,:) = [];

        fprintf('Disabled feature: %s - remaining features for evaluation %d\n', RFE.sortedByRank.Name{cntNumFeatures,1}, cntNumFeatures-1);
            
    end

    if length(tmpRFEFeatureList) == 1
        RFE.sortedByRank.Name{1,1} = tmpRFEFeatureList{1};
        RFE.trainedClassifier{1,1} = [];
        RFE.featureSubsetNames{1,1} = tmpRFEFeatureList(1);
    else
        fprintf('noch mehr Eintr�ge �brig!!\n\n');
    end

end

