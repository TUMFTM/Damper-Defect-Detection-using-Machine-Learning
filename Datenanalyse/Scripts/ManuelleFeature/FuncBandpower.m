function FuncBandpower = FuncBandpower(data,f1,f2,opts)
% Calculates bandpower for frequency band between f1 and f2
% Change boundary values in function "FeatureExtraction"

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    
    FuncBandpower.(fieldsData{cntSens}) = bandpower(data.(fieldsData{cntSens})',opts.fs,[f1 f2])';
    
end

end