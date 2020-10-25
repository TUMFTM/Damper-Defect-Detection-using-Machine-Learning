function FuncCrossCorr = FuncCrossCorr(data,maxLags,opts)
% Calculates cross correlation from two different sensor time signals
% Output is 
% - the maximum absolute cross correlation coefficient based on the
%   smoothed signal of the cross correlation coefficients within "maxLags"
% - the actual lag for the coefficient above

%% Aquire data
fieldsData = fields(data);

for cntSens=1 : size(fieldsData,1)
    for cntCorrSens = (cntSens+1) : size(fieldsData,1)        
        for cntObs = 1 : size(data.(fieldsData{cntSens}),1)
            [coeff, lags] = xcorr(data.(fieldsData{cntSens})(cntObs,:),data.(fieldsData{cntCorrSens})(cntObs,:),maxLags);
            coeff_smoothed = smooth(lags,coeff);
            [maxAbsCoeff, idxMaxAbsCoeff] = max(abs(coeff_smoothed));
            FuncCrossCorr.([fieldsData{cntSens},'_',fieldsData{cntCorrSens}]).maxAbsCoeff(cntObs,:) = maxAbsCoeff;
            FuncCrossCorr.([fieldsData{cntSens},'_',fieldsData{cntCorrSens}]).maxAbsCoeffLag(cntObs,:) = lags(idxMaxAbsCoeff);
        end
    end
end

end