function FuncCrest = FuncCrest(data,opts)
% Returns crest factor for time signal

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncCrest.(fieldsData{cntSens}) = peak2rms(data.(fieldsData{cntSens}),2);
end

end