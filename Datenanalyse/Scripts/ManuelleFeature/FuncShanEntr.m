function FuncShanEntr = FuncShanEntr(data,opts)
% Calculates shannon entropie from time signal

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    
    for cntObs = 1 : size(data.(fieldsData{cntSens}),1)
        
        FuncShanEntr.(fieldsData{cntSens})(cntObs,1) = wentropy(data.(fieldsData{cntSens})(cntObs,:),'shannon');
        
    end
    
end

end