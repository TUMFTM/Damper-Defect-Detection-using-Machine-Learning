function RFE = performRFEusingAccuracy(featuresTraining, varargin)
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
    
    if find(strcmp(varargin,'classifierOpts'))
        classifierOpts = varargin{find(strcmp(varargin,'classifierOpts'))+1};
    else
        classifierOpts = setOptsClassifier(classifierType);
    end
    classifierOpts.optimizeHyperparameters = '';    % disable classifier hyperparameter optimization
    
    if find(strcmp(varargin,'evaluateFeatureBlocks'))
        evaluateFeatureBlocks = varargin{find(strcmp(varargin,'evaluateFeatureBlocks'))+1};
    else
        evaluateFeatureBlocks = 0;
        fitPosterior = true;
    end
    
    if find(strcmp(varargin,'evaluateSignalBlocks'))
        evaluateSignalBlocks = varargin{find(strcmp(varargin,'evaluateSignalBlocks'))+1};
    else
        evaluateSignalBlocks = 0;
        fitPosterior = true;
    end
    
    if find(strcmp(varargin,'fieldsData'))
        fieldsData = varargin{find(strcmp(varargin,'fieldsData'))+1};
    else
        fieldsData = 0;
    end
    
    if find(strcmp(varargin,'featuresValidation'))
        featuresValidation = varargin{find(strcmp(varargin,'featuresValidation'))+1};
    else
        % Split training data to actual training data and validation data
        n = size(featuresTraining.data,1);
        tf = false(n,1);
        tf(1:round(0.8*n)) = true;
        tf = tf(randperm(n));
        [featuresTraining, featuresValidation] = splitFeatureStruct(featuresTraining, tf);
        featuresValidation.featureNames = featuresValidation.data.Properties.VariableNames;
    end
    
    %% Acutal calculation
    listOfAllFeatureNamesToBeEvaluated = featuresTraining.data.Properties.VariableNames';
    
    if evaluateFeatureBlocks
        % Get unique feature function blocks by extracting signal names out
        % of the feature names
        featureBlockNames = extractBefore(listOfAllFeatureNamesToBeEvaluated, '_');
        listOfAllFeatureNamesToBeEvaluated = unique(featureBlockNames,'stable');
        
    elseif evaluateSignalBlocks
        
        allFeatureNames = featuresTraining.data.Properties.VariableNames';

        % delete last number with underscore (e.g delete '_1')
        endingNumbers = cell(size(allFeatureNames));
        for cntFeat = 1 : length(allFeatureNames)
            tmpName = split(allFeatureNames(cntFeat), '_');
            endingNumbers(cntFeat) = tmpName(end);
        end
        % Get unique feature names without last number
        uniqueEndingNumbers = unique(endingNumbers, 'stable');
        
        
        % delete last number with underscore (e.g delete '_1')
        featNamesWoNumbers = cell(size(allFeatureNames));
        for cntFeat = 1 : length(allFeatureNames)
            tmpName = split(allFeatureNames(cntFeat), '_');
            featNamesWoNumbers(cntFeat) = join(tmpName(1:end-1),'_');
        end
        % Get unique feature names without last number
        uniqueFeatNamesWoNumbers = unique(featNamesWoNumbers, 'stable');
        
        if length(allFeatureNames)/length(uniqueEndingNumbers) == length(uniqueFeatNamesWoNumbers)
            signalNames = uniqueFeatNamesWoNumbers;
        else
            
            % find individual feature blocks
            featureBlockNames = extractBefore(allFeatureNames, '_');
            uniqueFeatureBlockNames = unique(featureBlockNames,'stable');

            % find a feature block with as much single features as signals
            numberOfFeaturesPerFeatureBlock = zeros(length(uniqueFeatureBlockNames), 1);
            for cntFeat = 1 : length(uniqueFeatureBlockNames)
                numberOfFeaturesPerFeatureBlock(cntFeat) = sum(contains(allFeatureNames,uniqueFeatureBlockNames{cntFeat}));
            end
            [numSignals, idxIndividualFeatureBlock] = min(numberOfFeaturesPerFeatureBlock);
            featureNamesOfIndividualFeatureBlock = allFeatureNames(contains(allFeatureNames, featureBlockNames{idxIndividualFeatureBlock}));

            % delete last number with underscore (e.g delete '_1')
            for cntSignals = 1 : numSignals
                tmpName = split(featureNamesOfIndividualFeatureBlock(cntSignals), '_');
                featureNamesOfIndividualFeatureBlock(cntSignals) = join(tmpName(1:end-1),'_');
            end

            % Identify name of feature without signal name
            numberOfIdenticCharacters = zeros(numSignals,1);
            for cntSignals = 1 : numSignals
                numberOfIdenticCharacters(cntSignals) = length(featureNamesOfIndividualFeatureBlock{cntSignals});
                tmpNames = featureNamesOfIndividualFeatureBlock;
                tmpNames(cntSignals) = [];
                for cntChar = 1 : length(featureNamesOfIndividualFeatureBlock{cntSignals})
                    charIsInOtherFeatNames = strncmp(tmpNames, featureNamesOfIndividualFeatureBlock{cntSignals}, cntChar);
                    if min(charIsInOtherFeatNames)
                        numberOfIdenticCharacters(cntSignals) = cntChar;
                    else
                        break;
                    end
                end
            end
            featureName = featureNamesOfIndividualFeatureBlock{1,1}(1:min(numberOfIdenticCharacters));

            % Get signal names
            signalNames = extractAfter(featureNamesOfIndividualFeatureBlock, featureName);
            
        end
        
        listOfAllFeatureNamesToBeEvaluated = signalNames;

    else
        % Single Feature Analysis
        % disable Crossvalidation for single feature analysis
        classifierOpts.validation = 0.2;
        fitPosterior = false;
    end

    numFeatures = length(listOfAllFeatureNamesToBeEvaluated);

    RFE = struct();
    RFE.sortedByRank = struct();
    RFE.sortedByRank.Name = cell(numFeatures,1);
    RFE.featureSubsetNames = cell(numFeatures,1);
    
    listOfRemainingSingleFeatures = featuresTraining.data.Properties.VariableNames';
    listOfRemainingFeatureNamesToBeEvaluated = listOfAllFeatureNamesToBeEvaluated;
    
    % Calculate classifier with all features before for-loop
    RFE.trainedClassifier{numFeatures,1} = trainClassifier(featuresTraining, classifierOpts, 'TextOutput', 1, 'selectedFeatures', listOfRemainingSingleFeatures, 'FitPosterior', fitPosterior);
    if iscell(RFE.trainedClassifier{numFeatures,1})
        RFE.featureSubsetNames{numFeatures,1} = RFE.trainedClassifier{numFeatures,1}{1,1}.ExpandedPredictorNames';
    else
        RFE.featureSubsetNames{numFeatures,1} = RFE.trainedClassifier{numFeatures,1}.ExpandedPredictorNames';
    end
    
    for cntNumFeatures = numFeatures : -1 : 2

        trainedClassifier = cell(cntNumFeatures,1);
        accuracy = cell(cntNumFeatures,1);
        parfor cntInnerNumFeatures = 1 : cntNumFeatures
        
            innerFeatureList = listOfRemainingSingleFeatures;
            
            idxInnerSingleFeaturesToBeDeleted = contains(listOfRemainingSingleFeatures, listOfRemainingFeatureNamesToBeEvaluated{cntInnerNumFeatures});
            innerFeatureList(idxInnerSingleFeaturesToBeDeleted) = [];
            
            trainedClassifier{cntInnerNumFeatures,1} = ...
                trainClassifier(featuresTraining, classifierOpts, 'TextOutput', 1, 'selectedFeatures', innerFeatureList, 'FitPosterior', fitPosterior);

            predictedClass = predictClassifier(trainedClassifier{cntInnerNumFeatures,1}, featuresValidation.data);
            if length(predictedClass) == length(trainedClassifier{cntInnerNumFeatures,1})
                misclassification = zeros(size(predictedClass));
                for cntCV = 1 : length(predictedClass)
                    misclassification(cntCV) = sum(~strcmp(featuresValidation.Label,predictedClass{cntCV}))/length(featuresValidation.Label);
                end
            else
                misclassification = sum(~strcmp(featuresValidation.Label,predictedClass))/length(featuresValidation.Label);
            end

            accuracy{cntInnerNumFeatures,1} = 100 * (1 - misclassification);

            fprintf('Overall remaining features: %d, Disabled feature %d: %s. Mean Accuracy: %.2f +- %.2f %%\n', ...
                cntNumFeatures, cntInnerNumFeatures, listOfRemainingFeatureNamesToBeEvaluated{cntInnerNumFeatures}, mean(accuracy{cntInnerNumFeatures,1}), std(accuracy{cntInnerNumFeatures,1}));
            
        end
        
        [del_max,del_idx] = max(cellfun(@mean,accuracy));
        RFE.sortedByRank.Name{cntNumFeatures,1} = listOfRemainingFeatureNamesToBeEvaluated{del_idx};
        
        RFE.trainedClassifier{cntNumFeatures-1,1} = trainedClassifier{del_idx};
        if iscell(RFE.trainedClassifier{cntNumFeatures-1,1})
            RFE.featureSubsetNames{cntNumFeatures-1,1} = RFE.trainedClassifier{cntNumFeatures-1,1}{1,1}.ExpandedPredictorNames';
        else
            RFE.featureSubsetNames{cntNumFeatures-1,1} = RFE.trainedClassifier{cntNumFeatures-1,1}.ExpandedPredictorNames';
        end
        
        listOfRemainingFeatureNamesToBeEvaluated(del_idx,:) = [];
        
        idxInnerSingleFeaturesToBeDeleted = contains(listOfRemainingSingleFeatures, RFE.sortedByRank.Name{cntNumFeatures,1});
        listOfRemainingSingleFeatures(idxInnerSingleFeaturesToBeDeleted) = [];

        fprintf('Max. test accuracy: %.2f %%. Feature name: %s - remaining features %d\n', del_max, RFE.sortedByRank.Name{cntNumFeatures,1}, cntNumFeatures-1);
            
    end
    
    if length(listOfRemainingFeatureNamesToBeEvaluated) == 1
        RFE.sortedByRank.Name{1,1} = listOfRemainingFeatureNamesToBeEvaluated{1};
    end

end

