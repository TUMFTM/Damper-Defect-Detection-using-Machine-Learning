function FuncRange = FuncRange(data,opts)
% Calculates range of structured data

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncRange.(fieldsData{cntSens}) = range(data.(fieldsData{cntSens}),2);
end

end