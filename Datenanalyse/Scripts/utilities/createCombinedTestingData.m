function [combinedTestset, combinedTrainingset] = createCombinedTestingData(allFeatureSets, varargin)
%CREATECOMBINEDTESTINGDATA Summary of this function goes here
%   Detailed explanation goes here

    if find(strcmp(varargin,'featuresTraining'))
        combinedTrainingset = varargin{find(strcmp(varargin,'featuresTraining'))+1};
    else
        combinedTrainingset = [];
    end
    
    numberOfInliers = 125;
    numberOfOutliers = 125;
    
    combinedTestset = [];
    
    for cntFeatSets = 1 : length(allFeatureSets)
        
        uniqueClassNames = unique(allFeatureSets{cntFeatSets}.Prop.labelIsolation);
        classNameInlier = uniqueClassNames{contains(uniqueClassNames, {'passive'})};
        featureSetInlier = reduceFeaturesToSpecificClass(allFeatureSets{cntFeatSets},classNameInlier);
        
        [inlierToBeAddedToCombinedTestSet, inlierToBeAddedToCombinedTrainingSet] = splitFeatureStruct(featureSetInlier, [], 'splitTrainingData', numberOfInliers);
%         [inlierToBeAddedToCombinedTrainingSet, ~] = splitFeatureStruct(inlierRest, [], 'splitTrainingData', numberOfInliers);
        
        if isempty(combinedTestset)
            combinedTestset = inlierToBeAddedToCombinedTestSet;
        else
            combinedTestset = mergeStructs(combinedTestset, inlierToBeAddedToCombinedTestSet, 1);
        end
        
        if isempty(combinedTrainingset)
            combinedTrainingset = inlierToBeAddedToCombinedTrainingSet;
        else
            combinedTrainingset = mergeStructs(combinedTrainingset, inlierToBeAddedToCombinedTrainingSet, 1);
        end
        
        classNameOutlier = uniqueClassNames(~contains(uniqueClassNames, {'passive'}));
        for cntClass = 1 : length(classNameOutlier)
            featureSetOutlier = reduceFeaturesToSpecificClass(allFeatureSets{cntFeatSets},classNameOutlier{cntClass});
            
            [OutlierToBeAddedToCombinedTestSet, OutlierToBeAddedToCombinedTrainingSet] = splitFeatureStruct(featureSetOutlier, [], 'splitTrainingData', round(numberOfOutliers/length(classNameOutlier)));
%             [OutlierToBeAddedToCombinedTrainingSet, ~] = splitFeatureStruct(restOutlier, [], 'splitTrainingData', round(numberOfOutliers/length(classNameOutlier)));
            
            combinedTestset = mergeStructs(combinedTestset, OutlierToBeAddedToCombinedTestSet, 1);
            combinedTrainingset = mergeStructs(combinedTrainingset, OutlierToBeAddedToCombinedTrainingSet, 1);
        end
    end
    
    combinedTestset.featureNames = combinedTestset.featureNames(1,:);
    combinedTestset.uniqueClasses = unique(combinedTestset.Prop.labelIsolation);
    combinedTestset.labelAsMatrix = generateLabelAsMatrix(combinedTestset, 'uniqueClasses', combinedTestset.uniqueClasses);
    
    combinedTrainingset.featureNames = combinedTrainingset.featureNames(1,:);
    combinedTrainingset.uniqueClasses = unique(combinedTrainingset.Prop.labelIsolation);
    combinedTrainingset.labelAsMatrix = generateLabelAsMatrix(combinedTrainingset, 'uniqueClasses', combinedTrainingset.uniqueClasses);
end

