function FuncNorm = FuncNorm(data,opts)
% Calculates euclidean-norm of time signal

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncNorm.(fieldsData{cntSens}) = vecnorm(data.(fieldsData{cntSens}),2,2);
end

end