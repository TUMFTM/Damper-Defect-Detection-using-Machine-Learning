function [FisherScore] = calcFisherScore(featuresData, labelAsMatrix)
%CALCFISHERSCORE Summary of this function goes here
%   Detailed explanation goes here

    %% Process observations per class
    numClasses = size(labelAsMatrix,2);

    [numObs, numFeatures] = size(featuresData);

    %% Compute Fisher-Score
    if istable(featuresData)
        featuresData = table2array(featuresData);
    end
    
    muOfEachFeatureOverAllObservations = mean(featuresData);   % mu^k
    
    muOfEachFeatureOverAllObservationsOfOneClass = zeros(numClasses, numFeatures);
    varOfEachFeatureOverAllObservationsOfOneClass = zeros(numClasses, numFeatures);
    numObservationsOfOneClass = zeros(numClasses, 1);
    for cntClass = 1 : numClasses
        muOfEachFeatureOverAllObservationsOfOneClass(cntClass,:) = mean(featuresData(labelAsMatrix(:,cntClass)==1,:),1);     % mu_i^k
        varOfEachFeatureOverAllObservationsOfOneClass(cntClass,:) = var(featuresData(labelAsMatrix(:,cntClass)==1,:),1);     % sigma_i^k^2
        numObservationsOfOneClass(cntClass,1) = sum(labelAsMatrix(:,cntClass)==1);    % n_i
    end
    
    FisherScore = sum(numObservationsOfOneClass .* (muOfEachFeatureOverAllObservationsOfOneClass - muOfEachFeatureOverAllObservations).^2,1) ./ ...
        sum(numObservationsOfOneClass .* varOfEachFeatureOverAllObservationsOfOneClass,1);
    FisherScore(isnan(FisherScore)) = 0;
    
    FisherScore = FisherScore';

end

