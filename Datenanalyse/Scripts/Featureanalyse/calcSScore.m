function [SScore] = calcSScore(featuresData, labelAsMatrix)
%CALCFISHERSCORE Summary of this function goes here
%   Detailed explanation goes here

    %% Process observations per class
    numClasses = size(labelAsMatrix,2);

    [numObs, numFeatures] = size(featuresData);

    %% Compute S-Score
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
    
    idx = 1;
    for cntClass1 = 1 : (numClasses-1)
        for cntClass2 = (cntClass1+1) : numClasses
            SScore(idx,:) = abs(muOfEachFeatureOverAllObservationsOfOneClass(cntClass1,:) - muOfEachFeatureOverAllObservationsOfOneClass(cntClass2,:)) ./ ...
                    (varOfEachFeatureOverAllObservationsOfOneClass(cntClass1,:) + varOfEachFeatureOverAllObservationsOfOneClass(cntClass2,:));
            idx = idx + 1;
        end
    end
    SScore(isnan(SScore)) = 0;
    SScore = mean(SScore,1);
    
    SScore = SScore';

end

