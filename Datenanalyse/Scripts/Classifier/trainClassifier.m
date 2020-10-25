function [trainedClassifier] = trainClassifier(featuresTraining, classifierOpts, varargin)
%TRAINCLASSIFIER2 Summary of this function goes here
%   Detailed explanation goes here

    if find(strcmp(varargin,'TextOutput'))
        TextOutput = varargin{find(strcmp(varargin,'TextOutput'))+1};
    else
        TextOutput = 1;
    end
    
    if find(strcmp(varargin,'selectedFeatures'))
        selectedFeatures = varargin{find(strcmp(varargin,'selectedFeatures'))+1};
    else
        selectedFeatures = featuresTraining.data.Properties.VariableNames;
    end
    
    if find(strcmp(varargin,'FitPosterior'))
        fitPosterior = varargin{find(strcmp(varargin,'FitPosterior'))+1};
    else
        fitPosterior = true;
    end

    switch classifierOpts.classifierType
        case 'SVM'
            trainedClassifier = trainSVM(featuresTraining,classifierOpts, 'TextOutput', TextOutput, 'selectedFeatures', selectedFeatures, 'FitPosterior', fitPosterior);
            try
                trainedClassifier = compact(trainedClassifier);
            end
            
        case 'Tree'
            trainedClassifier = trainEnsembleTree(featuresTraining,classifierOpts, 'TextOutput', TextOutput, 'selectedFeatures', selectedFeatures);
            
        case 'DiscriminantAnalysis'
            trainedClassifier = trainDiscriminantAnalysisClassifier(featuresTraining,classifierOpts);
            
        case 'kNN'
            trainedClassifier = trainkNN(featuresTraining,classifierOpts);
            
        otherwise
            warning('Unexpected classifier type')

    end


end

