function FuncCov = FuncCov(data,Fpass1,Fpass2,enable_filter,opts)
% Power spectral density (PSD) via Covariance method
% Input left (a) and right (b) boundary
%% Boundaries unfiltered data
% ACC_DATA: a1 = 10, b1 = 18 | a = 55, b = 90
% SPEED_DATA: a2 = 10, b2 = 18 | a = 55, b = 90
% Change boundary values in function "FeatureExtraction"
%% Boundaries filtered data
% ACC_DATA: a1 = 3, b1 = 7 | a = 15, b = 36
% SPEED_DATA: a2 = 3, b2 = 6 | a = 15, b = 30
% Change boundary values in function "FeatureExtraction"

%% Calculation

fieldsData = fields(data);

% Generate Filter and set parameters
if enable_filter
    res_cov = 8;
    varFilter = generateFilter(Fpass1,Fpass2);
else
    res_cov = 4;
end

for cntSens = 1 : size(fieldsData,1)

    % Filter time signals
    if enable_filter
        tmp_data = filter(varFilter,data.(fieldsData{cntSens}),2);
    else
        tmp_data = data.(opts.fieldsData{cntSens});
    end
    
    [tmp_DATA_cov, tmp_freq] = pcov(tmp_data',res_cov,[],opts.fs);
    tmp_DATA_cov = tmp_DATA_cov';
    ind = tmp_freq>=Fpass1 & tmp_freq<= Fpass2;
    [tmp_peak, peak_locfreq2] = max(tmp_DATA_cov(:,ind),[],2);
    tmp_freq_reduced = tmp_freq(ind);
    tmp_loc = tmp_freq_reduced(peak_locfreq2);
    
    FuncCov.peak.(fieldsData{cntSens}) = tmp_peak;
    FuncCov.loc.(fieldsData{cntSens}) = tmp_loc;

end
 
end
