function analyzeHistogramOfAllProperties(features)
%ANALYZEALLPROPERTIES Summary of this function goes here
%   Detailed explanation goes here

    fieldsProp = fields(features.Prop);
    
    for cntField = 1 : length(fieldsProp)
        if (isnumeric(features.Prop.(fieldsProp{cntField})(1,1))) && (size(features.Prop.(fieldsProp{cntField}),2)==1)
            histogramAnalysisContVar(features.Prop.Posterior, features.labelAsMatrix, features.Prop.(fieldsProp{cntField}), (fieldsProp{cntField}));
        elseif iscell(features.Prop.(fieldsProp{cntField}))
            histogramAnalysisCatVar(features.Prop.Posterior, features.labelAsMatrix, features.Prop.(fieldsProp{cntField}), (fieldsProp{cntField}));
        end
    end

end

