function FuncAbsEnergyTime = FuncAbsEnergyTime(data,opts)
% Calculates normalized absolute energy of time signal

%% Aquire data

fieldsData = fields(data);

for cntSens=1:size(fieldsData,1)
    FuncAbsEnergyTime.(fieldsData{cntSens}) = sum(data.(fieldsData{cntSens}).^2,2);
end

end