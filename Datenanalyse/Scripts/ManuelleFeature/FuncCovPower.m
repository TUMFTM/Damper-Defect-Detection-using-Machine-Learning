function FuncCovPower = FuncCovPower(data,freq_lower,freq_upper,interval,opts)
% Calculates band energy of fregency interval +-interval via Covariance method
% Input left (freq_lower) and right (freq_upper) boundary
%% Boundaries unfiltered data
% Change boundary values in function "FeatureExtraction"

%% Calculation
fieldsData = fields(data);

res_cov = 4;

for cntSens = 1 : size(fieldsData,1)

    % Calculation of peaks and location
    tmp_data = data.(fieldsData{cntSens});
    [tmp_DATA_cov, tmp_freq] = pcov(tmp_data',res_cov,[],opts.fs);
    tmp_DATA_cov = tmp_DATA_cov';
    ind = tmp_freq>=freq_lower & tmp_freq<= freq_upper;
    [tmp_peak, peak_locfreq2] = max(tmp_DATA_cov(:,ind),[],2);
    tmp_freq_reduced = tmp_freq(ind);
    tmp_loc = tmp_freq_reduced(peak_locfreq2);
    
    % man könnte noch nach einem Weg suchen, um die Bandpower-Funktion als
    % Matrixoperation durchzuführen. Das Problem sind die unterschiedlichen
    % Auswerteintervalle.
    FuncCovPower.(fieldsData{cntSens}) = zeros(size(tmp_DATA_cov,1),1);
    for cntObs=1:size(tmp_DATA_cov,1)
        FuncCovPower.(fieldsData{cntSens})(cntObs) = bandpower(tmp_data(cntObs,:)',opts.fs,[tmp_loc(cntObs)-interval tmp_loc(cntObs)+interval])';
    end
    
end

end

