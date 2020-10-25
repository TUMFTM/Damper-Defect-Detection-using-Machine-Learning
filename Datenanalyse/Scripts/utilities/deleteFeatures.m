function table_out = deleteFeatures(table_in, featureNames)
%REDUCEFEATURES
%Script for reducing the DATA_classification table (or any other table)
%from the field names featureNames

    table_out = table_in;
    for cntFeature = 1 : length(featureNames)
        fieldsIn = fields(table_out);
        idx = contains(fieldsIn,featureNames{cntFeature});
        table_out(:,idx) = [];
    end
    

end

