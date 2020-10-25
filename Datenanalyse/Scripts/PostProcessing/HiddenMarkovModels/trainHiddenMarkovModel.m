function [mu, sigma, TRANS] = trainHiddenMarkovModel(labelAsMatrix, varargin)
%TRAINHIDDENMARKOVMODEL Summary of this function goes here
%   Detailed explanation goes here

    % set Transition matrix to identity matrix
    thresholdTrans = 0.95;

    %% Einlesen der übergebenen Größen
    if find(strcmp(varargin,'features'))
        features = varargin{find(strcmp(varargin,'features'))+1};
    else
        features = 0;
    end
    
    if find(strcmp(varargin,'trainedClassifier'))
        trainedClassifier = varargin{find(strcmp(varargin,'trainedClassifier'))+1};
    else
        trainedClassifier = 0;
    end
    
    if find(strcmp(varargin,'aprioriProbabilityMatrix'))
        aprioriProbabilityMatrix = varargin{find(strcmp(varargin,'aprioriProbabilityMatrix'))+1};
    else
        [~, aprioriProbabilityMatrix] = predictClassifier(trainedClassifier, features);
    end

    %% Calculation
    % Generate number of classes
    n_classes = size(aprioriProbabilityMatrix,2);

    GMModel = cell(n_classes,1);
    mu = zeros(n_classes);
    sigma = zeros(n_classes,n_classes,n_classes);
    for cntClass = 1 : n_classes
        idx = labelAsMatrix(:,cntClass)==1;
        GMModel{cntClass} = fitgmdist(aprioriProbabilityMatrix(idx,:),1,'RegularizationValue',0.1,'CovarianceType','diagonal');
        mu(cntClass,:) = GMModel{cntClass}.mu;
        if size(GMModel{cntClass}.Sigma,1) == 1
            sigma(:,:,cntClass) = diag(GMModel{cntClass}.Sigma);
        else
            sigma(:,:,cntClass) = GMModel{cntClass}.Sigma;
        end
    end
    
    TRANS = thresholdTrans * eye(n_classes) + (1-thresholdTrans) / (n_classes-1) * (ones(n_classes)-eye(n_classes));

end

