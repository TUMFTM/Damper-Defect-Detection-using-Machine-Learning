function testdataTesting = testBinaryClassifier(trainedClassifier, testdataTesting, featureNamesToBeDeleted, labelMapping, renameFeatureNamesMapping, featureExtractionHandle, opts)
%CLASSIFYINDIVIDUALCLASS Summary of this function goes here
%   Detailed explanation goes here

    %     labelMapping = {'fromLabel', 'toLabel';...}

    % Delete some features
    featuresClassifier = testdataTesting.features.featureNames(~contains(testdataTesting.features.featureNames,featureNamesToBeDeleted));
    testdataTesting.features.data = reduceFeatures(testdataTesting.features.data, featuresClassifier);
    testdataTesting.features.featureNames = testdataTesting.features.data.Properties.VariableNames;
    
    % Rename features
    if ~isempty(renameFeatureNamesMapping)
        for cntFeat = 1 : size(renameFeatureNamesMapping,1)
            testdataTesting.features = renameFeatureNames(testdataTesting.features, renameFeatureNamesMapping{cntFeat,1}, renameFeatureNamesMapping{cntFeat,2});
        end
    end
    
    % Map labels to new values
    for cntLabel = 1 : size(labelMapping,1)
        testdataTesting.features.Prop.labelIsolation(strcmp(testdataTesting.features.Prop.labelIsolation,labelMapping{cntLabel,1})) = labelMapping(cntLabel,2);
        testdataTesting.data.Prop.labelIsolation(strcmp(testdataTesting.data.Prop.labelIsolation,labelMapping{cntLabel,1})) = labelMapping(cntLabel,2);
    end
    testdataTesting.features.Label = testdataTesting.features.Prop.labelIsolation;
    
    % Overwrite unique classes
    testdataTesting.features.uniqueClasses = getUniqueClasses(testdataTesting.data);
    
    % Generate labelAsMatrix because of label conversion
    testdataTesting.features.labelAsMatrix = generateLabelAsMatrix(testdataTesting.features);
    
    if isfield(testdataTesting.features, 'dataAsArray')
        testdataTesting.features.dataAsArray = table2array(testdataTesting.features.data);
    end
    
    % Performance on Testing data of each binary classifier
    if ~isempty(trainedClassifier)
        testdataTesting = testClassifier(trainedClassifier, featureExtractionHandle, opts, testdataTesting.features);
    end
    
    testdataTesting.featureNamesToBeDeleted = featureNamesToBeDeleted;
    testdataTesting.labelMapping = labelMapping;
    testdataTesting.renameFeatureNamesMapping = renameFeatureNamesMapping;

end

