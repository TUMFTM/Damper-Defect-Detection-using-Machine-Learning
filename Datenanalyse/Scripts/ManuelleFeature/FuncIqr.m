function FuncIqr = FuncIqr(data,opts)
% Builds interquartile range of time signal

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncIqr.(fieldsData{cntSens}) = iqr(data.(fieldsData{cntSens}),2);
end

end