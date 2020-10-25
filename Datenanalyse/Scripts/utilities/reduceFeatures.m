function table_out = reduceFeatures(table_in, featureNames)
% reduceFeatures-Script
% Script for reducing the DATA_classification table (or any other table)
% from the field names featureNames
% The names in featureNames are searched in table_in. If they exist
% precisley, those features are selected. If the don't exist, but exist in
% slightly different form 
% e.g featureNames = "FuncARfit_SPEED_FL_3" and "FuncARfit_SPEED_FL"
% exists, than this whole feature block ("FuncARfit_SPEED_FL") is selected.
% This leads to more selected featuers than specified in featureNames
% It makes sense to use the function rearrangeFeatureTable befor using
% the reduceFeatures-function

    fprintf('\nReducing to specified feature blocks...');
    
    table_out = table();
    
    for cntFeature = 1 : length(featureNames)
        fieldsIn = fields(table_in);
        [matchingFeatures, ~, endFeatures] = regexp([featureNames{cntFeature},'*'],fieldsIn,'match','start','end');
        idx = find(~cellfun(@isempty,matchingFeatures));
        
        cntIdx = 1;
        lengthFeatureName = length(featureNames{cntFeature});
        while max(size(idx)) > 1
            if endFeatures{idx(cntIdx)} ~= lengthFeatureName
                idx(cntIdx) = [];
            else
                cntIdx = cntIdx + 1;
            end
        end

        foundFeature = matchingFeatures{idx(1)}{1};
        if ~strcmp(foundFeature,featureNames{cntFeature})
            fprintf('Looking for %s, found %s\n', featureNames{cntFeature}, foundFeature);
        end
        table_out.(foundFeature) = table_in.(foundFeature);
    end
    
    fprintf('Reduced to %d feature blocks with %d features', size(table_out,2), size(table2array(table_out),2));

    % Add label
    if isfield(table_in,'Label')
        table_out.Label = table_in.Label;
    end
    
end

