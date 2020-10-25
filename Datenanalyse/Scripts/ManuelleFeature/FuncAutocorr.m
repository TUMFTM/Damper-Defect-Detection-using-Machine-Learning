function FuncAutocorr = FuncAutocorr(data,order,opts)
% Calculates parameters of autocorrelation function

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncAutocorr.(fieldsData{cntSens}) = zeros(size(data.(fieldsData{cntSens}),1), order);
    tmp = zeros(size(data.(fieldsData{cntSens}),1), order+1);
    for cntObs = 1 : size(data.(fieldsData{cntSens}),1)
        tmp(cntObs,:) = autocorr(data.(fieldsData{cntSens})(cntObs,:), order);
    end
    FuncAutocorr.(fieldsData{cntSens}) = tmp(:,2:end);
end

end