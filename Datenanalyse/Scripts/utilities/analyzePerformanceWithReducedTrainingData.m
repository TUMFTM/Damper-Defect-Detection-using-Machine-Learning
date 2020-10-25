function [datasizeAnalysis] = analyzePerformanceWithReducedTrainingData(featuresTraining, varargin)
%ANALYZEPERFORMANCEWITHREDUCEDTRAININGDATA Summary of this function goes here
%   Detailed explanation goes here

    if find(strcmp(varargin,'classifierType'))
        classifierType = varargin{find(strcmp(varargin,'classifierType'))+1};
    else
        classifierType = 'SVM';
    end

    if find(strcmp(varargin,'classifierOpts'))
        classifierOpts = varargin{find(strcmp(varargin,'classifierOpts'))+1};
    else
        classifierOpts = setOptsClassifier(classifierType);
    end
    
%     nameFeaturesTesting = inputname(2);
    
    vecRelSizeOfTrainingData = [0.1 : 0.1 : 1];
    vecAbsSizeOfTrainingData = round(vecRelSizeOfTrainingData * size(featuresTraining.data,1));
    sizeVecRelSize = length(vecRelSizeOfTrainingData);
    
    datasizeAnalysis = struct();
    tmpClassifier = cell(sizeVecRelSize,1); % needed for parfor-loop
    datasizeAnalysis.relSizeFeaturesTraining = vecRelSizeOfTrainingData;
    datasizeAnalysis.absSizeFeaturesTraining = vecAbsSizeOfTrainingData;
%     datasizeAnalysis.(nameFeaturesTesting).accuracy = zeros(sizeVecRelSize,1);
    
    parfor cntSize = 1 : length(vecRelSizeOfTrainingData)
        
        n = size(featuresTraining.data,1);
        tf = false(n,1);
        tf(1:vecAbsSizeOfTrainingData(cntSize)) = true;
        tf = tf(randperm(n));
        
        featuresTrainingRedSize = splitFeatureStruct(featuresTraining, tf);
        
        tmpClassifier{cntSize,1} = trainClassifier(featuresTrainingRedSize, classifierOpts, 'TextOutput', 1);

%         [~, predictedProbability] = predictClassifier(datasizeAnalysis.classifier{cntSize,1}, featuresTraining.data);
% 
%         [misclassification,~,~,~] = confusion(featuresTraining.labelAsMatrix', predictedProbability');
%         datasizeAnalysis.(nameFeaturesTesting).accuracy(cntSize,1) = 100 * (1 - misclassification);
    end
    
    datasizeAnalysis.trainedClassifier = tmpClassifier;

end

