function FuncKurtosis = FuncKurtosis(data,opts)
% Builds kurtosis of structured data

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncKurtosis.(fieldsData{cntSens}) = kurtosis(data.(fieldsData{cntSens}),[],2);
end

end