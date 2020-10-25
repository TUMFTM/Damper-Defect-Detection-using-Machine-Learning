function FuncAbsEnergyFreq = FuncAbsEnergyFreq(data,opts)
% Calculates normalized absolute energy of fft

    fieldsData = fields(data);

    for cntSens = 1 : size(fieldsData,1)
        tmp_data = fft(data.(fieldsData{cntSens}),[],2);
        FuncAbsEnergyFreq.(fieldsData{cntSens}) = sum(tmp_data.*conj(tmp_data),2);
    end

end