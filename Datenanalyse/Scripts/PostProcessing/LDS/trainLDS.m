function [LDSmodel] = trainLDS(valNovScore,valBatch, valObsID)
%TRAINLDS Summary of this function goes here
%   Detailed explanation goes here
    
%     valNovScore = testData.val.features.Prop.Posterior{1};
%     valBatch = testData.val.features.Prop.batch;
%     valObsID = testData.val.features.Prop.observationID;

    [valObsID_sort, idx_sort] = sort(valObsID);
    valBatch_sorted = valBatch(idx_sort);
    valNovScore_sorted = valNovScore(idx_sort);

    uniqueBatch = unique(valBatch_sorted);
    maxObsPerBatch = 0;
    for cntBatch = 1 : length(uniqueBatch)
        currBatch = uniqueBatch(cntBatch);
        obsPerBatch = sum(valBatch==currBatch);
        maxObsPerBatch = max(maxObsPerBatch, obsPerBatch);
    end
    valNovScore_sorted_matrix = NaN(maxObsPerBatch, length(uniqueBatch));
    S_batch = zeros(1,length(uniqueBatch));
    for cntBatch = 1 : length(uniqueBatch)
        currBatch = uniqueBatch(cntBatch);
        idxCurrBatch = valBatch_sorted==currBatch;
        obsPerBatch = sum(valBatch==currBatch);
        valNovScore_sorted_matrix(1:obsPerBatch,cntBatch) = valNovScore_sorted(valBatch_sorted==currBatch);
        if sum(idxCurrBatch)>=1
            S_batch(cntBatch) = var(valNovScore_sorted(valBatch_sorted==currBatch));
        else
            S_batch(cntBatch) = NaN;
        end
    end

    LDSmodel = struct();
    LDSmodel.A = 1;
    LDSmodel.C = 1;
    LDSmodel.S = mean(S_batch(~isnan(S_batch)));
    LDSmodel.G = 0.05*LDSmodel.S;
    
end

