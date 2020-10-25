function [posteriorNovScore, posteriorNovScoreCov] = calcLDS(aprioriNovScore, aprioriBatch, aprioriObsID, LDSmodel)
%CALCLDS Summary of this function goes here
%   Detailed explanation goes here

    [~, idx_sort] = sort(aprioriObsID);
    aprioriNovScore_sorted = aprioriNovScore(idx_sort);
    testBatch_sorted = aprioriBatch(idx_sort);

    posteriorNovScore = zeros(size(aprioriNovScore_sorted));
    posteriorNovScoreCov = zeros(size(aprioriNovScore_sorted));

    uniqueBatch = unique(testBatch_sorted);
    for cntBatch = 1 : length(uniqueBatch)
        currBatch = uniqueBatch(cntBatch);
        idxCurrBatch = testBatch_sorted==currBatch;
        LDSmodel.mu0 = mean(aprioriNovScore_sorted(idxCurrBatch));
        LDSmodel.P0 = var(aprioriNovScore_sorted(idxCurrBatch));
        [nu, U, llh, Ezz, Ezy] = kalmanSmoother(LDSmodel, aprioriNovScore_sorted(idxCurrBatch)');
        posteriorNovScore(idx_sort(idxCurrBatch)) = nu';
        posteriorNovScoreCov(idx_sort(idxCurrBatch)) = squeeze(U);
    end

end

