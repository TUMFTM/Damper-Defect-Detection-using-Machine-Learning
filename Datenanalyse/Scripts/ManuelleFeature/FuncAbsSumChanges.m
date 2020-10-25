function FuncAbsSumChanges = FuncAbsSumChanges(data,opts)
% Calculates absolute sum of changes of time signal

%% Aquire data
fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncAbsSumChanges.(fieldsData{cntSens}) = sum(abs(diff(data.(fieldsData{cntSens}),[],2)),2);
end

end