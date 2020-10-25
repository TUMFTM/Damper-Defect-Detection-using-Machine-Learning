function [posteriorProbabilityMatrix, posteriorPredictedClassSequence] = calcMeanProbabilityForEachMeasurement(aprioriProbability, vecMeasurements, varargin)
%CALCULATEMEANPROBABILITY Summary of this function goes here
%   Detailed explanation goes here

    %% Einlesen der übergebenen Größen
%     if find(strcmp(varargin,'aprioriProbability'))
%         aprioriProbability = varargin{find(strcmp(varargin,'aprioriProbability'))+1};
%     else
%         aprioriProbability = features.Prop.Posterior;
%     end

    posteriorProbabilityMatrix = zeros(size(aprioriProbability));
    
    for cntBatch = 1 : max(vecMeasurements)
        
        % Identify relevant observations of current measurement
        idxMeasurement = logical(vecMeasurements == cntBatch);
        
        % Calculate posterior Probability of the current measurement
        posteriorProbabilityMatrix(idxMeasurement,:) = ones(sum(idxMeasurement),1).*mean(aprioriProbability(idxMeasurement,:));
        
    end

    % Generate Sequence of posterior predicted classes
    [~,posteriorPredictedClassSequence] = max(posteriorProbabilityMatrix,[],2);
    
end

