function FuncComplTimeSeries = FuncComplTimeSeries(data,lag,opts)
% Calculates efficient complexity-invariant distance for time series
% Function gives a measure of complexity
% Change lag in function "FeatureExtraction"

%% Aquire data

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    FuncComplTimeSeries.(fieldsData{cntSens}) = sqrt(sum(diff(data.(fieldsData{cntSens}),lag,2).^2,2));
end

end