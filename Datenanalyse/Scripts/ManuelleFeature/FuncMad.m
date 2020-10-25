function FuncMad = FuncMad(data,opts)
% Calculates mean absolute deviation of structured data

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncMad.(fieldsData{cntSens}) = mad(data.(fieldsData{cntSens}),[],2);
end

end