function classifierOpts = setOptsClassifier(classifierType, varargin)
%SETOPTSDISCRIMINANTCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here

    classifierOpts = struct;

    switch classifierType
        case 'SVM'
            classifierOpts.template = templateSVM(...
                'KernelFunction', 'linear', ...
                'PolynomialOrder', [], ...
                'KernelScale', 'auto', ...
                'BoxConstraint', 1, ...    % 45 f�r lineare SVM und DD2 mit man. Feat.
                'Standardize', true);
            classifierOpts.Coding = 'onevsall';
            classifierOpts.optimizeHyperparameters = '';
            classifierOpts.MaxObjectiveEvaluations = 30;
            classifierOpts.validation = 5;  % >1 for kFold validation, <1 for percentage of holdOut-validation
        
        case 'Tree'
            classifierOpts.template = templateTree(...
                'Surrogate','on',...
                'MaxNumSplits', 10,...
                'MinLeafSize', 5);
            classifierOpts.optimizeHyperparameters = '';
            classifierOpts.validation = 0.3;
            classifierOpts.Method = 'AdaBoostM2';
            classifierOpts.LearnRate = 0.94523;

        case 'DiscriminantAnalysis'
            classifierOpts.SaveMemory = 'off';
            classifierOpts.DiscrimType = 'linear'; % linear, diaglinear, pseudolinear, quadratic, diagquadratic, pseudoquadratic
            classifierOpts.Gamma = 0;  % Amount of regularization (range [0,1])
            classifierOpts.optimizeHyperparameters = 'all';   % none, auto, all or {'parameter1';'parameter2',...}
            classifierOpts.validation = 0.3;
            
        case 'kNN'
            classifierOpts.Distance = 'Euclidean';
            classifierOpts.Exponent = [];
            classifierOpts.NumNeighbors = 101;
            classifierOpts.DistanceWeight = 'Equal';
            classifierOpts.Standardize = true;
            classifierOpts.optimizeHyperparameters = '';
            classifierOpts.validation = 0.3;

        otherwise
            warning('Unexpected classifier type')

    end

    % Copy classifier type (SVM, kNN etc.) to classifierOpts
    classifierOpts.classifierType = classifierType;
    
    % Overwrite parameters if they are explicitly set
    if find(strcmp(varargin,'optimizeHyperparameters'))
        classifierOpts.optimizeHyperparameters = varargin{find(strcmp(varargin,'optimizeHyperparameters'))+1};
    end
    
end

