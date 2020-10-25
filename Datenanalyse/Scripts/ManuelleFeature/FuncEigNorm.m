function FuncEigNorm = FuncEigNorm(data,filterEig,filterRef,freqRef,freqEig,opts)
% Builds Pseudospectrum estimate via Eigenvector method and compares
% values of low frequency (a) maxima to their reference location (b)
% Boundarys for ACC_DATA: a = 3.5, b = 16 [Hz]
% Boundarys for SPEED_DATA: a = 3.5, b = 16 [Hz]
% Change boundary values in function "FeatureExtraction"
% Additional "condensed" feature from acc & speed data generated in "FeatureExtraction"

%% Calculation
fieldsData = fields(data);

filterEigFreq = generateFilter(filterEig(1),filterEig(2));
filterRefFreq = generateFilter(filterRef(1),filterRef(2));

for cntSens = 1 : size(fieldsData,1)

    tmpDataEig = filter(filterEigFreq,data.(fieldsData{cntSens}),2);
    tmpDataRef = filter(filterRefFreq,data.(fieldsData{cntSens}),2);
    
    for cntObs = 1 : size(tmpDataEig,1)
    
        [tmpSEig(cntObs,:), tmp_freq] = peig(tmpDataEig(cntObs,:),8,[],opts.fs);
        tmpSEig(cntObs,:) = smooth(tmpSEig(cntObs,:));
        
        [tmpSRef(cntObs,:), tmp_freq] = peig(tmpDataEig(cntObs,:),8,[],opts.fs);
        tmpSRef(cntObs,:) = smooth(tmpSRef(cntObs,:));
        
    end
    
    if size(freqRef) == size(fieldsData,1)
        freqEvalRef = freqRef(cntSens);
    else
        freqEvalRef = freqRef;
    end
    if size(freqEig) == size(fieldsData,1)
        freqEvalEig = freqEig(cntSens);
    else
        freqEvalEig = freqEig;
    end
    peakRef = tmpSRef(:,find(tmp_freq>=freqEvalRef,1,'first'));
    peakEig = tmpSEig(:,find(tmp_freq>=freqEvalEig,1,'first'));
    FuncEigNorm.(fieldsData{cntSens}) = peakRef./peakEig;
    
end

end
