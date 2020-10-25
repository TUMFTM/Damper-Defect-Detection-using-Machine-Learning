function FuncMean = FuncMean(data,opts)
% Builds mean of structured data

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncMean.(fieldsData{cntSens}) = mean(data.(fieldsData{cntSens}),2);
end

end