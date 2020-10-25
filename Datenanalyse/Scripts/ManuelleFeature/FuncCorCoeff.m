function FuncCorCoeff = FuncCorCoeff(data,opts)
% Calculates correlation coefficient from two different sensor time signals
% Output is first secondary diagonal element of correlation matrix

%% Aquire data
fieldsData = fields(data);

for cntSens=1 : size(fieldsData,1)
    for cntCorrSens = (cntSens+1) : size(fieldsData,1)
        for cntObs = 1 : size(data.(fieldsData{cntSens}),1)
            FuncCorCoeff.([fieldsData{cntSens},'_',fieldsData{cntCorrSens}])(cntObs,1) = ...
                diag(corrcoef(data.(fieldsData{cntSens})(cntObs,:),data.(fieldsData{cntCorrSens})(cntObs,:)),1);
        end
    end
end

end