function structTest = importVAEResult(pathResultFolder, origData, type)
%ANALYZECNNPERFORMANCEF Summary of this function goes here
%   Detailed explanation goes here
% clear;clc;

% Example call:
% load workspace from FFT
% testDD2MassCNN = analyzeCNNPerformance(testDD2Mass, 'W:\Projekte\Fahrwerkdiagnose\Paper\CNN\results\2019_02_22_15h_42_Robustness_mass_17h_50_depth1D_e5_k16_cv5\d_3_addc_0\fold_0');

    structTest = struct();
%     ctrl.pathResultFolder = 'W:\Projekte\Fahrwerkdiagnose\Paper\CNN\results\2019_02_22_12h_35\fold_0';
%     ctrl.pathResultFolder = 'W:\Projekte\Fahrwerkdiagnose\Datenanalyse\CNN\results\2020_03_06_15h_46\addc_2_e_015_poolingsize_016_cv5\fold_0';
    ctrl.pathResultFolder = pathResultFolder;

%     predictionLabels = csvread(fullfile(ctrl.pathResultFolder, ['PredictionLabels',type,'.csv']));
    predictionLabels = NaN;
    predictionScores = csvread(fullfile(ctrl.pathResultFolder, ['PredictionScores',type,'.csv']));
    trueLabels = csvread(fullfile(ctrl.pathResultFolder, ['TrueLabels',type,'.csv']));
    index = csvread(fullfile(ctrl.pathResultFolder, ['Index',type,'.csv']));
    
    if min(index)==0
        index = index+1;
    end
    
    if size(trueLabels,2)==1
        % check if index must be increased by 1
        % python starts counting from 0 but Matlab from 1, however, index is
        % loaded from the Matlab export
        labelsOrigData = origData.Prop.LabelID(index);
        if length(unique(trueLabels))==2
            labelsOrigData(labelsOrigData==0) = min(trueLabels);
            labelsOrigData(~labelsOrigData==0) = max(trueLabels);
            if ~isequal(labelsOrigData, trueLabels)
                index = index+1;
            end
        end
    else
        trueLabels = trueLabels*[1:size(trueLabels,2)]';
    end

    % initialize Prop-structure
    fieldsOrigDataProp = fields(origData.Prop);
    f = [fieldsOrigDataProp'];
    f{2,1} = {[]};
    structTest.features.Prop = struct(f{:});
    if isfield(structTest.features.Prop,'predictedClass') && isnan(predictionLabels)
        structTest.features.Prop = rmfield(structTest.features.Prop,'predictedClass');
        fieldsOrigDataProp = fields(structTest.features.Prop);
    end
    
    structTest.features.Prop.Posterior = predictionScores;
%     structTest.features.Prop.predictedClass = predictedClass;
%     structTest.features.Prop.Label = trueLabelName;
    structTest.features.Prop.observationID = index;
    if exist('classes','var')
        structTest.features.uniqueClasses = classes;
    else
        structTest.features.uniqueClasses = unique(origData.Prop.labelIsolation,'stable');
    end

    for cntField = 1 : length(fieldsOrigDataProp)
        if isempty(structTest.features.Prop.(fieldsOrigDataProp{cntField}))
            for cntObs = 1 : length(structTest.features.Prop.observationID)
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

    structTest.features.labelAsMatrix = generateLabelAsMatrix(structTest.features, 'uniqueClasses', structTest.features.uniqueClasses);
    structTest.features.Label = structTest.features.Prop.labelIsolation;
    structTest.data.Prop.Posterior = structTest.features.Prop.Posterior;

    structTest.classifier = ['see log.txt in ', ctrl.pathResultFolder];

end

