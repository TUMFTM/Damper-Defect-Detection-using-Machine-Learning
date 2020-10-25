function FStdv = FuncStdv(data,opts)
% Calculates standard deviation of structured data

%% Aquire data

fieldsData = fields(data);

for cntSens=1:size(fieldsData,1)
    FStdv.(fieldsData{cntSens}) = std(data.(fieldsData{cntSens}),0,2);
end

end