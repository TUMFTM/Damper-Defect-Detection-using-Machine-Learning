function [posteriorProbabilityMatrix, posteriorPredictedClassSequence] = calcWeightedMeanProbabilityForEachMeasurement(aprioriProbability, vecMeasurements, varargin)
%CALCULATEMEANPROBABILITY Summary of this function goes here
%   Detailed explanation goes here

    posteriorProbabilityMatrix = zeros(size(aprioriProbability));
    
    for cntBatch = 1 : max(vecMeasurements)
        
        % Identify relevant observations of current measurement
        idxMeasurement = logical(vecMeasurements == cntBatch);
        
        % Calculate 'weight' for weighted mean apriori probability
        % 'weight' has same length as aprioirPredictedProbabilityMatrix
        % sum(weight) = 1
        maxAprioirProbability = max(aprioriProbability,[],2);
        idxMaxProbApriorProb = bsxfun(@eq, aprioriProbability, maxAprioirProbability);
        tmp = aprioriProbability;
        tmp(idxMaxProbApriorProb) = 0;
        maxProb2 = max(tmp,[],2);
        weight = zeros(size(aprioriProbability,1),1);
        weight(idxMeasurement) = (maxAprioirProbability(idxMeasurement) - maxProb2(idxMeasurement)) ./ sum(maxAprioirProbability(idxMeasurement) - maxProb2(idxMeasurement));
        
        % Calculate posterior weighted mean probability of the current measurement
        posteriorProbabilityMatrix(idxMeasurement,:) = ones(sum(idxMeasurement),1)*(weight(idxMeasurement)' * aprioriProbability(idxMeasurement,:));
        
    end

    % Generate Sequence of posterior predicted classes
    [~,posteriorPredictedClassSequence] = max(posteriorProbabilityMatrix,[],2);
    
end

