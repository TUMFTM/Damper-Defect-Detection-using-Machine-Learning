function FuncCovNorm = FuncCovNorm(data,filterFreqLower,filterFreqUpper,res_cov,opts)
% Builds power spectral density (PSD) via Covariance method and compares
% values of low frequency (a) maxima to their reference location (b)
% Boundarys for ACC_DATA: a = 3.5, b = 16 [Hz]
% Boundarys for SPEED_DATA: a = 3.5, b = 16 [Hz]
% Change boundary values in function "FeatureExtraction"
% Additional "condensed" feature from acc & speed data generated in "FeatureExtraction"

%% Aquire data

fieldsData = fields(data);

% filterEigFreq = generateFilter(10,15);
% filterRefFreq = generateFilter(2.5,5);

for cntSens = 1 : size(fieldsData,1)

    % Redefinition of frequency intervals for peak search
    if size(filterFreqLower) == size(fieldsData,1)
        tmpFilterFreqLower = filterFreqLower(cntSens,:);
    else
        tmpFilterFreqLower = filterFreqLower;
    end
    if size(filterFreqUpper) == size(fieldsData,1)
        tmpFilterFreqUpper = filterFreqUpper(cntSens,:);
    else
        tmpFilterFreqUpper = filterFreqUpper;
    end
    filterLower = generateFilter(tmpFilterFreqLower(1),tmpFilterFreqLower(2));
    filterUpper = generateFilter(tmpFilterFreqUpper(1),tmpFilterFreqUpper(2));
    
    tmpDataFiltLower = filter(filterLower,data.(fieldsData{cntSens}),2);
    tmpDataFiltUpper = filter(filterUpper,data.(fieldsData{cntSens}),2);
    
    % Calculation of PSD
    [PxxLower, PxxFreqLower] = pcov(tmpDataFiltLower',res_cov,[],opts.fs);
    PxxLower = PxxLower';
    [PxxUpper, PxxFreqUpper] = pcov(tmpDataFiltUpper',res_cov,[],opts.fs);
    PxxUpper = PxxUpper';
    
    % Find peaks in specified frequency intervals
    indLower = PxxFreqLower>=tmpFilterFreqLower(1) & PxxFreqLower<= tmpFilterFreqLower(2);
    indUpper = PxxFreqUpper>=tmpFilterFreqUpper(1) & PxxFreqUpper<= tmpFilterFreqUpper(2);
    [PeakLower, tmpPeakLowerFreq] = max(PxxLower(:,indLower),[],2);
    [PeakUpper, tmpPeakUpperFreq] = max(PxxUpper(:,indUpper),[],2);
    tmpFreqLowerReduced = PxxFreqLower(indLower);
    tmpFreqUpperReduced = PxxFreqUpper(indUpper);
%     PeakLowerFreq = tmpFreqLowerReduced(tmpPeakLowerFreq);
%     PeakUpperFreq = tmpFreqUpperReduced(tmpPeakUpperFreq);

    FuncCovNorm.(fieldsData{cntSens}) = PeakLower./PeakUpper;

end

end
