function [posteriorProbabilityMatrix, posteriorPredictedClassSequence] = calcMovMeanProbabilityForEachMeasurement(aprioriProbability, vecMeasurements)
%CALCULATEMEANPROBABILITY Summary of this function goes here
%   Detailed explanation goes here

    posteriorProbabilityMatrix = zeros(size(aprioriProbability));
    
    for cntBatch = 1 : max(vecMeasurements)
        
        % Identify relevant observations of current measurement
        idxMeasurement = logical(vecMeasurements == cntBatch);
        
        % Calculate posterior moving mean probability of the current measurement
        posteriorProbabilityMatrix(idxMeasurement,:) = movmean(aprioriProbability(idxMeasurement,:),[1000000,0]);
        
    end

    % Generate Sequence of posterior predicted classes
    [~,posteriorPredictedClassSequence] = max(posteriorProbabilityMatrix,[],2);
    
end

