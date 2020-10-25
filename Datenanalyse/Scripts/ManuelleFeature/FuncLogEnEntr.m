function FuncLogEnEntr = FuncLogEnEntr(data,opts)
% Calculates log energy entropie from time signal

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    
    for cntObs = 1 : size(data.(fieldsData{cntSens}),1)
        
        FuncLogEnEntr.(fieldsData{cntSens})(cntObs,1) = wentropy(data.(fieldsData{cntSens})(cntObs,:),'log energy');
        
    end
    
end

end