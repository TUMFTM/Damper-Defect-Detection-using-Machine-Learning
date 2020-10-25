function [datasizeAnalysis] = analyzePerformanceWithReducedTrainingDataRepLearn(dataTraining, opts, ctrl, varargin)
%ANALYZEPERFORMANCEWITHREDUCEDTRAININGDATA Summary of this function goes here
%   Detailed explanation goes here

    if find(strcmp(varargin,'classifierType'))
        classifierType = varargin{find(strcmp(varargin,'classifierType'))+1};
    else
        classifierType = 'SVM';
    end
    
    if find(strcmp(varargin,'featureExtractionHandle'))
        featureExtractionHandle = varargin{find(strcmp(varargin,'featureExtractionHandle'))+1};
    end
    
    if find(strcmp(varargin,'autoencoderOpts'))
        autoencoderOpts = varargin{find(strcmp(varargin,'autoencoderOpts'))+1};
        autoencoderOpts.optimizeHyperparameter = 0;
    else
        autoencoderOpts = 0;
    end
    
    if find(strcmp(varargin,'sparsefilterOpts'))
        sparsefilterOpts = varargin{find(strcmp(varargin,'sparsefilterOpts'))+1};
    else
        sparsefilterOpts = 0;
    end

    if find(strcmp(varargin,'classifierOpts'))
        classifierOpts = varargin{find(strcmp(varargin,'classifierOpts'))+1};
    else
        classifierOpts = setOptsClassifier(classifierType);
    end
    
    vecRelSizeOfTrainingData = [0.1 : 0.1 : 1];
    vecAbsSizeOfTrainingData = round(vecRelSizeOfTrainingData * size(dataTraining.(opts.fieldsData{1}),1));
    sizeVecRelSize = length(vecRelSizeOfTrainingData);
    
    datasizeAnalysis = struct();
    tmpClassifier = cell(sizeVecRelSize,1); % needed for parfor-loop
    
    tmpfeatureExtractionHandle = cell(sizeVecRelSize,1); % needed for parfor-loop
    if exist('featureExtractionHandle')
        for cntSize = 1 : length(vecRelSizeOfTrainingData)
            tmpfeatureExtractionHandle{cntSize,1} = featureExtractionHandle;
        end
    end
    datasizeAnalysis.relSizeFeaturesTraining = vecRelSizeOfTrainingData;
    datasizeAnalysis.absSizeFeaturesTraining = vecAbsSizeOfTrainingData;
    removedTrainingSamples = cell(sizeVecRelSize,1);
    
    parfor cntSize = 1 : length(vecRelSizeOfTrainingData)
        
        n = size(dataTraining.(opts.fieldsData{1}),1);
        tf = false(n,1);
        tf(1:vecAbsSizeOfTrainingData(cntSize)) = true;
        tf = tf(randperm(n));
        removedTrainingSamples{cntSize,1} = find(~tf);
        dataTrainingReduced = removeCorruptedObservations(dataTraining, removedTrainingSamples{cntSize,1});

        if strcmp(ctrl.featureType, 'Autoencoder')
            [tmpmdlAutoencoder, tmpautoencoderOpts] = initAndTrainAutoencoder(dataTrainingReduced,opts,autoencoderOpts); % Preprocessing of data and training of Autoencoder
            tmpfeatureExtractionHandle{cntSize,1} = @(x)initAndTestAutoencoder(x,tmpmdlAutoencoder,opts,tmpautoencoderOpts);
        elseif strcmp(ctrl.featureType, 'Sparsefilter')
            [tmpmdlSparsefilter, tmpsparsefilterOpts] = initAndTrainSparsefilter(dataTrainingReduced,opts,sparsefilterOpts); % Preprocessing of data and training of Sparsefilter
            tmpfeatureExtractionHandle{cntSize,1} = @(x)initAndTestSparsefilter(x,tmpmdlSparsefilter,opts,tmpsparsefilterOpts);     % Define function handle for feature generation with Sparsefilter       
        end
        featuresTraining = generateFeatureStruct(dataTrainingReduced, tmpfeatureExtractionHandle{cntSize,1});
        tmpClassifier{cntSize,1} = trainClassifier(featuresTraining, classifierOpts, 'TextOutput', 1);

    end
    
    datasizeAnalysis.trainedClassifier = tmpClassifier;
    datasizeAnalysis.featureExtractionHandle = tmpfeatureExtractionHandle;
    datasizeAnalysis.removedTrainingSamples = removedTrainingSamples;

end

