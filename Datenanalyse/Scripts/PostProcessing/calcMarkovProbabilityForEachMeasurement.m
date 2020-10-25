function [posteriorProbabilityMatrix, posteriorPredictedClassSequence] = calcMarkovProbabilityForEachMeasurement(mu, sigma, TRANS, aprioriProbability, vecMeasurement)
%CALCULATEMEANPROBABILITY Summary of this function goes here
%   Detailed explanation goes here

    posteriorProbabilityMatrix = zeros(size(aprioriProbability,1),size(TRANS,2));
    n_classes = size(aprioriProbability,2);
    
    batch_vec = unique(vecMeasurement);
    for idxBatch = 1 : length(batch_vec)
        
        cntBatch = batch_vec(idxBatch);

        % Identify relevant observations of current measurement
        idxMeasurement = logical(vecMeasurement == cntBatch);

        % Set initial state probability
        if sum(idxMeasurement)>1
            p = mean(aprioriProbability(idxMeasurement,:))';
        else
            p = 1/n_classes * ones(n_classes,1);
        end

        data = aprioriProbability(idxMeasurement,:)';
        prior = p;
        mixmat = p;
        obslik = mixgauss_prob(data, mu, sigma, mixmat);
%         [path, delta, prob] = viterbi_path(prior, TRANS, obslik);
        [alpha, beta, gamma, ~] = fwdback(prior, TRANS, obslik);
        posteriorProbabilityMatrix(idxMeasurement,:) = gamma';
%         [~,tmp] = max(gamma',[],2);
%         path = path';
%         if ~isequal(path,tmp)
%             fprintf('In cntBatch=%d with sum(idx...)=%d from idx %d to %d\n', cntBatch, sum(idxMeasurement), find(idxMeasurement,1,'first'), find(idxMeasurement,1,'last'))
%         end
        
    end

    % Generate Sequence of posterior predicted classes
    [~,posteriorPredictedClassSequence] = max(posteriorProbabilityMatrix,[],2);
    
end

