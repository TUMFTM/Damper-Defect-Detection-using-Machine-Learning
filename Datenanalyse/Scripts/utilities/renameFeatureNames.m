function featureStructOut = renameFeatureNames(featureStructIn, oldPattern, newPattern)
%RENAMEFEATURENAMES Summary of this function goes here
%   Detailed explanation goes here
    
    featureStructOut = featureStructIn;
    featureStructOut.data.Properties.VariableNames = strrep(featureStructIn.data.Properties.VariableNames, oldPattern, newPattern);
    featureStructOut.featureNames = featureStructOut.data.Properties.VariableNames;
    
end

