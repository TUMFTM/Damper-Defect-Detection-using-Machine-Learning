function FuncMin = FuncMin(data,opts)
% Builds minimum value of structured data

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncMin.(fieldsData{cntSens}) = min(data.(fieldsData{cntSens}),[],2);
end

end

