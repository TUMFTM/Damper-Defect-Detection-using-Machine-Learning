function FuncMsWT = FuncMsWT(data,win_size,opts)
% Builds multiscale wavelet transformation and extracts several features
% Input: Structured data (X), analysis window size (win_size) & sensor number (z)
% Output: Energy, variance, stdv, waveform length & entropy for every sample window
% Y1: Waveform length, Y2: Energy Y3: Variance, Y4: Stdv, Y5: Entropy
% Change win_size in function "FeatureExtraction"

%% MsWT
fieldsData = fields(data);
N = 10;

for cntSens = 1 : size(fieldsData,1)

    for cntObs = 1 : size(data.(fieldsData{cntSens}),1)
        
        tmp_X = data.(fieldsData{cntSens})(cntObs,:);
        l = length(tmp_X);
        n_win = floor(l/win_size);
        tmp_X = tmp_X(1:n_win*win_size);
        data_feat = detrend(reshape(tmp_X,[win_size,n_win]));
        dec = mdwtdec('col',data_feat,N,'db4');
        [coeffs,lengths] = wdec2cl(dec,'all');
        coeffs = coeffs';
        lengths = lengths';
        level = length(lengths)-2;
        n_coeffs = size(coeffs,2);
        COEFFS = coeffs.^2;
        energy = sum(COEFFS,2);
        idx = (energy>0);
        percent_energy(idx,:) = 100*COEFFS(idx,:)./energy(idx,ones(1,n_coeffs));

        a = 1;
        b = 1;

        for k=1:level+1
            n_coeffs = lengths(k);
            b  = a+n_coeffs-1;
            energy(:,k) = mean(percent_energy(:,a:b),2);
            variance(:,k) = var(percent_energy(:,a:b),[],2);
            stdv(:,k) = std(percent_energy(:,a:b),[],2);
            wf_length(:,k) = sum(abs(diff(percent_energy(:,a:b)').^2))';
            percent_energy(:,a:b) = percent_energy(:,a:b)./repmat(sum(percent_energy(:,a:b),2),1,size(percent_energy(:,a:b),2));
            entropy(:,k) = -sum(percent_energy(:,a:b).*log(percent_energy(:,a:b)),2)./size(percent_energy(:,a:b),2);
            a = b + 1;
        end

        FuncMsWT.wf_length.(fieldsData{cntSens})(cntObs,:) = mean(log1p(wf_length));
        FuncMsWT.energy.(fieldsData{cntSens})(cntObs,:) = mean(log1p(energy));
        FuncMsWT.variance.(fieldsData{cntSens})(cntObs,:) = mean(log1p(variance));
        FuncMsWT.stdv.(fieldsData{cntSens})(cntObs,:) = mean(log1p(stdv));
        FuncMsWT.entropy.(fieldsData{cntSens})(cntObs,:) = mean(entropy);
        
    end
    
end

end

