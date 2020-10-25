function FuncSampEntr = FuncSampEntr(data,dim,r,opts)
% Computes the sample entropy of a time series
% Input: 2 < dim < 4 | 0.1 < r < 0.25
% Change dim & r in function "FeatureExtraction"

%% Calculation

fieldsData = fields(data);

for cntSens = 1 : size(fieldsData,1)
            
    tmp_data = data.(fieldsData{cntSens});
    tmp_data = (tmp_data - mean(tmp_data,2)) ./ std(tmp_data,[],2);
    N = size(tmp_data,2);
    rStdv = r * std(tmp_data,[],2);
    
    for cntObs = 1 : size(data.(fieldsData{cntSens}),1)
        
        tmp_X = tmp_data(cntObs,:);
        
        for k = 1 : 2
            
            M = dim + k - 1;
            
            prob = zeros(1,N-M+1);
            comp_series = zeros(M,N-M+1);
            for l = 1:M
                comp_series(l,:) = tmp_X(l:N-M+l);
            end
    
            for l = 1:N-M+1
                tmp = max(abs(comp_series - repmat(comp_series(:,l),1,N-M+1)));
                idx = (tmp <= rStdv(cntObs));
                count(l) = (sum(idx)-1);
            end
            count = count/(N-M-1);
            tmp_Y(k) = mean(count);
            
        end
        
        FuncSampEntr.(fieldsData{cntSens})(cntObs,1) = log(tmp_Y(1)/tmp_Y(2));
        
    end
    
end

end