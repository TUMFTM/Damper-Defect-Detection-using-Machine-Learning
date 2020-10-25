function FuncSkewness = FuncSkewness(data,opts)
% Builds skewness of structured data

%% Aquire data
fieldsData = fields(data);

for cntSens=1:size(fieldsData,1)
    FuncSkewness.(fieldsData{cntSens}) = skewness(data.(fieldsData{cntSens}),1,2);
end

end