function FuncADFtest = FuncADFtest(data,opts)
% Performs augmented Dickey-Fuller test on structured data

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    for cntObs = 1 : size(data.(fieldsData{cntSens}),1)
        FuncADFtest.(fieldsData{cntSens})(cntObs,1) = adftest(data.(fieldsData{cntSens})(cntObs,:));
    end
end

end