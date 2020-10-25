function FuncRMS = FuncRMS(data,opts)
% Calculates root-mean-square-level of structured data

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncRMS.(fieldsData{cntSens}) = rms(data.(fieldsData{cntSens}),2);
end

end