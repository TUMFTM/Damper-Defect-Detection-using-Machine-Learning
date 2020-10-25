function FuncEig = FuncEig(data,Fpass1,Fpass2,enable_filter,opts)
% Pseudospectrum estimate via Eigenvector method
% Input left (a) and right (b) boundary
%% Boundaries unfiltered data
% ACC_DATA: a1 = 9, b1 = 16 | a = 46, b = 82
% SPEED_DATA: a2 = 9, b2 = 16 | a = 46, b = 82
% Change boundary values in function "FeatureExtraction"
%% Boundaries filtered data
% ACC_DATA: a1 = 3, b1 = 7 | a = 15, b = 36
% SPEED_DATA: a2 = 3, b2 = 6 | a = 15, b = 30
% Change boundary values in function "FeatureExtraction"

%% Calculation
fieldsData = fields(data);

% Generate Filter and set parameters
res_eig = 8;
if enable_filter
    varFilter = generateFilter(Fpass1,Fpass2);
end

for cntSens = 1 : size(fieldsData,1)

    % Filter time signals
    if enable_filter
        tmp_data = filter(varFilter,data.(fieldsData{cntSens}),2);
    else
        tmp_data = data.(fieldsData{cntSens});
    end
    
    for cntObs = 1 : size(tmp_data,1)
        [tmp_data_eig, tmp_data_freq] = peig(tmp_data(cntObs,:),res_eig,[],opts.fs);
        
        idx = tmp_data_freq>=Fpass1 & tmp_data_freq<= Fpass2;
        idx_tmp_peak = find(tmp_data_eig==max(tmp_data_eig(idx)),1);
        
        FuncEig.peak.(fieldsData{cntSens})(cntObs,1) = tmp_data_eig(idx_tmp_peak);
        FuncEig.loc.(fieldsData{cntSens})(cntObs,1) = tmp_data_freq(idx_tmp_peak);
    end

end

end

