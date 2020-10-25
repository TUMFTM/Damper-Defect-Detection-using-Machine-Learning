function FuncEigPower = FuncEigPower(data,freq_lower,freq_upper,interval,res_eig,opts)
% Calculates band energy of fregency interval +-interval via Eigenvector method
% Input left (a) and right (b) boundary
%% Boundaries unfiltered data
% ACC_DATA: a1 = 9, b1 = 16 | a = 46, b = 82
% SPEED_DATA: a2 = 9, b2 = 16 | a = 46, b = 82
% Change boundary values in function "FeatureExtraction"

%% Calculation

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)

    % Calculation of peaks and location
    tmp_data = data.(fieldsData{cntSens});

    FuncEigPower.(fieldsData{cntSens}) = zeros(size(tmp_data,1),1);
    
    for cntObs = 1 : size(tmp_data,1)
        [tmp_data_eig, tmp_data_freq] = peig(tmp_data(cntObs,:),res_eig,[],opts.fs);
        
        idx = tmp_data_freq>=freq_lower & tmp_data_freq<= freq_upper;
        idx_tmp_peak = find(tmp_data_eig==max(tmp_data_eig(idx)),1);
        
        tmp_freq = tmp_data_freq(idx_tmp_peak);

        % man könnte noch nach einem Weg suchen, um die Bandpower-Funktion als
        % Matrixoperation durchzuführen. Das Problem sind die unterschiedlichen
        % Auswerteintervalle.
        FuncEigPower.(fieldsData{cntSens})(cntObs) = bandpower(tmp_data(cntObs,:)',opts.fs,[tmp_freq-interval tmp_freq+interval])';

    end
    
end

end

