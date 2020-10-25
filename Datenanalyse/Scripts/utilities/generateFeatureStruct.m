function features = generateFeatureStruct(data, funHandleFeatureExtraction, varargin)
%GENERATEFEATURESTRUCT Summary of this function goes here
%   Detailed explanation goes here
    
    features = struct;
    features.data = funHandleFeatureExtraction(data);
    features.data = rearrangeFeatureTable(features.data);
    features.featureNames = features.data.Properties.VariableNames;
    
    % Export features to array for later usage in Python
    features.dataAsArray = table2array(features.data);
    
    % Check if unique classes of exisiting classifier is available
    if find(strcmp(varargin,'uniqueClasses'))
        uniqueClasses = varargin{find(strcmp(varargin,'uniqueClasses'))+1};
        features.labelAsMatrix = generateLabelAsMatrix(data, 'uniqueClasses', uniqueClasses);
        features.uniqueClasses = uniqueClasses;
    else
        features.labelAsMatrix = generateLabelAsMatrix(data);
        features.uniqueClasses = getUniqueClasses(data);
    end
    
    features.Prop = data.Prop;
    features.Label = data.Prop.labelIsolation;
    
end

