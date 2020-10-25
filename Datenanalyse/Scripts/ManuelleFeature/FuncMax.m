function FuncMax = FuncMax(data,opts)
% Builds maximum value of structured data

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncMax.(fieldsData{cntSens}) = max(data.(fieldsData{cntSens}),[],2);
end

end