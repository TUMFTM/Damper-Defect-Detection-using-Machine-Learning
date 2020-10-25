function structTest = importCNNResult(pathResultFolder, origData, type)
%ANALYZECNNPERFORMANCEF Summary of this function goes here
%   Detailed explanation goes here
% clear;clc;

    structTest = struct();

    ctrl.pathResultFolder = pathResultFolder;

    predictionLabels = csvread(fullfile(ctrl.pathResultFolder, ['PredictionLabels',type,'.csv']));
    predictionScores = csvread(fullfile(ctrl.pathResultFolder, ['PredictionScores',type,'.csv']));
    trueLabels = csvread(fullfile(ctrl.pathResultFolder, ['TrueLabels',type,'.csv']));
    index = csvread(fullfile(ctrl.pathResultFolder, ['Index',type,'.csv']));

    if size(trueLabels,2)==1
        if min(trueLabels)==0
            trueLabels = trueLabels+1;
        end
    else
        trueLabels = trueLabels*[1:size(trueLabels,2)]';
    end
    predictionLabels = predictionLabels+1;
 
    % check if index must be increased by 1
    % python starts counting from 0 but Matlab from 1, however, index is
    % loaded from the Matlab export
    labelsOrigData = origData.Prop.LabelID(index);
    uniqueLabelsOrigData = unique(labelsOrigData);
    uniqueTrueLabels = unique(trueLabels);
    labelsAreEqual = 1;
    for cntLabel = 1 : length(uniqueLabelsOrigData)
        idxCurrLabelOrigData = labelsOrigData == uniqueLabelsOrigData(cntLabel);
        idxCurrLabelTrueLabels = trueLabels == uniqueTrueLabels(cntLabel);
        if ~isequal(idxCurrLabelOrigData,idxCurrLabelTrueLabels)
            labelsAreEqual = 0;
        end
    end
    if labelsAreEqual==0
        index = index+1;
    end
    
    classes = unique(origData.Prop.labelIsolation,'stable');
    
    predictedClass = cell(size(predictionLabels));
    trueLabelName = cell(size(predictionLabels));
    for cntObs = 1 : size(predictionLabels,1)
        predictedClass{cntObs,1} = classes{predictionLabels(cntObs)};
        trueLabelName{cntObs,1} = classes{trueLabels(cntObs)};
    end

    % initialize Prop-structure
    fieldsOrigDataProp = fields(origData.Prop);
    f = [fieldsOrigDataProp'];
    f{2,1} = {[]};
    structTest.features.Prop = struct(f{:});
    
    structTest.features.Prop.Posterior = predictionScores;
    structTest.features.Prop.predictedClass = predictedClass;
    structTest.features.Prop.Label = trueLabelName;
    structTest.features.Prop.observationID = index;
    structTest.features.uniqueClasses = classes;
    structTest.features.labelAsMatrix = generateLabelAsMatrix(structTest.features, 'uniqueClasses', classes);

    for cntField = 1 : length(fieldsOrigDataProp)
        if isempty(structTest.features.Prop.(fieldsOrigDataProp{cntField}))
            for cntObs = 1 : length(structTest.features.Prop.predictedClass)
                idx = find(origData.Prop.observationID==structTest.features.Prop.observationID(cntObs));
                if iscell(origData.Prop.(fieldsOrigDataProp{cntField}))
                    structTest.features.Prop.(fieldsOrigDataProp{cntField}){cntObs,1} = origData.Prop.(fieldsOrigDataProp{cntField}){idx,1};
                elseif isdatetime(origData.Prop.(fieldsOrigDataProp{cntField}))
                    break
                else
                    structTest.features.Prop.(fieldsOrigDataProp{cntField})(cntObs,1) = origData.Prop.(fieldsOrigDataProp{cntField})(idx,1);
                end
            end
        end
    end

    structTest.data.Prop.Posterior = structTest.features.Prop.Posterior;
    structTest.data.Prop.predictedClass = structTest.features.Prop.predictedClass;

    structTest.classifier = ['see log.txt in ', ctrl.pathResultFolder];

end

