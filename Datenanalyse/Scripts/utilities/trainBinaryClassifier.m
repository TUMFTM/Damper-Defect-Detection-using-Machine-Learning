function [trainedClassifier, featuresTraining] = trainBinaryClassifier(featuresTrainingOrig,featureNamesToBeDeleted,labelMapping,renameFeatureNamesMapping,classifierOpts, ctrlTrainClassifier)
%CLASSIFYINDIVIDUALCLASS Summary of this function goes here
%   Detailed explanation goes here

    featuresTraining = featuresTrainingOrig;

    % Delete some features
    if ~isempty(featureNamesToBeDeleted)
        featuresClassifier = featuresTraining.featureNames(~contains(featuresTraining.featureNames,featureNamesToBeDeleted));
        featuresTraining.data = reduceFeatures(featuresTraining.data, featuresClassifier);
        featuresTraining.featureNames = featuresTraining.data.Properties.VariableNames;
    end
    
    % Rename features
    if ~isempty(renameFeatureNamesMapping)
        for cntFeat = 1 : size(renameFeatureNamesMapping,1)
            featuresTraining = renameFeatureNames(featuresTraining, renameFeatureNamesMapping{cntFeat,1}, renameFeatureNamesMapping{cntFeat,2});
        end
    end
    
    % Map labels to new values
    for cntLabel = 1 : size(labelMapping,1)
        featuresTraining.Prop.labelIsolation(strcmp(featuresTraining.Prop.labelIsolation,labelMapping{cntLabel,1})) = labelMapping(cntLabel,2);
    end
    featuresTraining.Label = featuresTraining.Prop.labelIsolation;
    
    % Overwrite unique classes
    featuresTraining.uniqueClasses = getUniqueClasses(featuresTraining);
    
    % Generate labelAsMatrix because of label conversion
    featuresTraining.labelAsMatrix = generateLabelAsMatrix(featuresTraining);
    
    if isfield(featuresTrainingOrig, 'dataAsArray')
        featuresTraining.dataAsArray = table2array(featuresTraining.data);
    end
    
    % Train Classifier
    if ctrlTrainClassifier
        trainedClassifier = trainClassifier(featuresTraining, classifierOpts);
    else
        trainedClassifier = [];
    end

end

