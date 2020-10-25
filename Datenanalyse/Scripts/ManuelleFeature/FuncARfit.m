function FuncARfit = FuncARfit(data,order,opts)
% Autoregressive model coefficients

%% AR

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    
    for cntObs = 1 : size(data.(fieldsData{cntSens}),1)
        
        AR = ar(iddata(data.(fieldsData{cntSens})(cntObs,:)',[],1/opts.fs),order);
        FuncARfit.(fieldsData{cntSens})(cntObs,:) = AR.A(:,2:end);
        
    end
    
end

end