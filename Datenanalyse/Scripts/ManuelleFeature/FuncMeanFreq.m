function FuncMeanFreq = FuncMeanFreq(data,opts)
% Calculates mean frequency of structured data

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    
    FuncMeanFreq.(fieldsData{cntSens}) = meanfreq(data.(fieldsData{cntSens})',opts.fs)';
    
end

end