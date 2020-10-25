function FuncDiscrPower = FuncDiscrPower(data,lag,opts)
% Calculates discrimination power of time series elements as a measure of non-linearity
% Input: lag = 5 | GET lag FROM PAPER [113]!!!
% Change lag in function "FeatureExtraction"
%% CTRL plot
ctrl = 0;
%% Aquire data
fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
    
    tmp_data = data.(fieldsData{cntSens});
    l = size(tmp_data,2);
    
    tmp_h = zeros(size(tmp_data,1),l-2*lag);
    for k=1:l-2*lag
        tmp_h(:,k) = tmp_data(:,k+2*lag).*tmp_data(:,k+lag).*tmp_data(:,k);
    end
    FuncDiscrPower.(fieldsData{cntSens}) = 1/(l-2*lag)*sum(tmp_h,2);
    
end

end