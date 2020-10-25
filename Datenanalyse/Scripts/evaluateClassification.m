clear;clc;
% Change to path or running script, to have correct relative paths
executionFilePath = strsplit(mfilename('fullpath'),filesep);
cd(strjoin(executionFilePath(1:end-1), filesep));
tic;

addpath(genpath(fullfile('..', 'Scripts')));

ctrl.saveData = 0;      % 1 = save data, 0 = don't save data
ctrl.pathToSave = fullfile('..','MachineLearning','ManuelleFeatures',[datestr(now,'yymmdd_HHMM')], filesep);    % path to save data
ctrl.matFileNameToSave = 'Workspace.mat';
ctrl.usePassiveForTraining = 0;     % use only data of "passive" setting as training data
ctrl.balanceDataset = 1;        % 0 = use all observations as training/test data; 1 = reduce the number of observations of each class to the number of observations with the fewest observations (recommended)
ctrl.featureType = 'manualFeatures';    % 'manualFeatures' or 'FFT' or 'Autoencoder' or 'Sparsefilter'
ctrl.classifierType = 'SVM';    % 'SVM' or 'kNN' or 'DiscriminantAnalysis' or 'Tree'
ctrl.trainClassifier = 1;       % 1 = train a classifier, 0 = don't train a classifier
ctrl.classifyEachTireIndividually = 1;  % classify data of each tire individually
ctrl.loadDataFromFile = '';         % if data shall be loaded from a mat-file specify name here, else leave it as ''
ctrl.loadWorkspaceFromFile = 0;     % 1 = load workspace from the specified file, 0 = don't load workspace from a mat-file
ctrl.performDataSizeAnalysis = 1;       % analysis of performance with reduced number of training samples for training process

ctrl.calcAllFeatures = 1;   % 1 = calc all manual features, 0 = calc only reduced set of manual features
ctrl.analyzeFisherScoreTraining = 1;        % calculate Fisher score
ctrl.analyzeFeatureImportanceUsingRandomForest = 0;     % analyze the feature importance using random forest
ctrl.reduceFeaturesBasedOnFisherScore = '';  % 'rank','score','' (empty)
ctrl.reduceFeaturesBasedOnFisherScoreTo = 100;    % depending on reduceBasedOnFisherScore reduce to top x ranks or reduce to top x FisherScore values
% Note: Reduction of features by Fisher Score is currently not supported 
% within the "testClassifier"-function below.

ctrl.performRFE = 1;        % perform a recursive feature elimination (RFE)
ctrl.RFEType = {'accuracySignalBlocks'; 'accuracyFeatureBlocks'; 'accuracy'};   % 'weights' or 'accuracy' or 'accuracyFeatureBlocks' or 'accuracySignalBlocks' or {'weights';'accuracy'} (for more)

%% Select Labeling file
filename = 'Label_DD2.xlsx';
foldername = ['..', filesep, 'Datensatz', filesep];
opts = setOptions('dataSource','Flexray');

%% Initialize folder for saving data
if ctrl.saveData
    % Check if folder for saving exists
    statusFolder = mkdir(ctrl.pathToSave);
    if ~statusFolder
        % Create folder iteratively
        currPath = pwd;
        for pathPart = split(ctrl.pathToSave)
            if ~exist(pathPart{1},'dir')
                mkdir(pathPart{1});
            end
            cd(pathPart{1});
        end
        cd(currPath);
    end
    
    % Activate logging of Command Window
    diary(fullfile(ctrl.pathToSave, [datestr(now, 'yymmdd_HHMM'), '_CommandWindowLog.txt']));
end

%% Preprocessing of Data
% Specify options of classifier
rng('default');     % used for auto kernel scale
classifierOpts = setOptsClassifier(ctrl.classifierType);

% Load data
if isempty(ctrl.loadDataFromFile)  % required as linux doesn't support mdf-files
    data = loadData(filename, foldername, opts);
else
    load(ctrl.loadDataFromFile);
end

% Some observations are corrupted (no automatic procedure)
if strcmp(filename,'Label_DD2.xlsx')
    data = removeCorruptedObservations(data, [[1637:1645], 1173, [1637:1645], 2753, 2757, 2758, 4880, [4963:4976], [5520:5534], 6104, [6697:6714], 9130, 9712, [11075:11092]]);
end

% Analyze balancedness of dataset
% analyseDataset(data,opts);
% if ischar(filename)
%     title(['Full Dataset ', filename]);
% else
%     title('Full Dataset');
% end
if ctrl.balanceDataset
    dataFull = data;
    [data, unbalancedRestData] = balanceDataset(data,opts);
%     analyseDataset(data,opts);
%     if ischar(filename)
%         title(['Balanced Dataset ', filename]);
%     else
%         title('Balanced Dataset');
%     end
%     analyseDataset(unbalancedRestData,opts);
%     if ischar(filename)
%         title(['unbalanced rest Dataset ', filename]);
%     else
%         title('unbalanced rest Dataset');
%     end
    clearvars unbalancedRestData;
end

% Split data
[dataTraining, dataTesting] = splitData(data, [], opts);
% analyseDataset(dataTraining,opts);
% title('Training Data');
% analyseDataset(dataTesting,opts);
% title('Testing Data');

%% Calculate Features
% Manual Features
if strcmp(ctrl.featureType, 'manualFeatures')
    featureExtractionHandle = @(x)featureExtraction(x,opts,'calcAllFeatures',ctrl.calcAllFeatures); % 1 for calculating all features

% FFT data points as features
elseif strcmp(ctrl.featureType, 'FFT')
    featureExtractionHandle = @(x)generateFFTDataAsTable(x,opts, 'windowlength', 128, 'windowstep', 64);
    
% Autoencoder Features
elseif strcmp(ctrl.featureType, 'Autoencoder')
    autoencoderOpts = setOptsAutoencoder('usePassiveForTraining', ctrl.usePassiveForTraining); % set options of Autoencoder
    [mdlAutoencoder, autoencoderOpts] = initAndTrainAutoencoder(dataTraining,opts,autoencoderOpts); % Preprocessing of data and training of Autoencoder
    featureExtractionHandle = @(x)initAndTestAutoencoder(x,mdlAutoencoder,opts,autoencoderOpts);    % Define function handle for feature generation with autoencoder
    
% Sparsefilter Features
elseif strcmp(ctrl.featureType, 'Sparsefilter')
    sparsefilterOpts = setOptsSparsefilter('usePassiveForTraining', ctrl.usePassiveForTraining);   % set options of Sparsefilter
    [mdlSparsefilter, sparsefilterOpts] = initAndTrainSparsefilter(dataTraining,opts,sparsefilterOpts); % Preprocessing of data and training of Sparsefilter
    featureExtractionHandle = @(x)initAndTestSparsefilter(x,mdlSparsefilter,opts,sparsefilterOpts);     % Define function handle for feature generation with Sparsefilter
end

%% Generate features of training data 
featuresTraining = generateFeatureStruct(dataTraining, featureExtractionHandle);

%% Analyze Fisher Score
if ctrl.analyzeFisherScoreTraining
    featuresTraining.FisherScore = analyzeFisherScore(featuresTraining);
    
    % Reduce Features Based on Fisher Score
    % Note: Reduction of features by Fisher Score is currently not supported 
    % within the "testClassifier"-function below.
    if strcmp(ctrl.reduceFeaturesBasedOnFisherScore,'rank')
        featuresTraining.dataAllFeatures = featuresTraining.data;
        featuresTraining.data = reduceFeatures(featuresTraining.data, featuresTraining.FisherScore.sortedByScore.Name(1:ctrl.reduceFeaturesBasedOnFisherScoreTo));
    elseif strcmp(ctrl.reduceFeaturesBasedOnFisherScore,'score')
        featuresTraining.dataAllFeatures = featuresTraining.data;
        featuresTraining.data = reduceFeatures(featuresTraining.data, featuresTraining.FisherScore.sortedByScore.Name(featuresTraining.FisherScore.sortedByScore.FisherScore>=ctrl.reduceFeaturesBasedOnFisherScoreTo));
    end
end

%% Train Classifier
if ctrl.trainClassifier
    trainedClassifier = trainClassifier(featuresTraining, classifierOpts);
else
    trainedClassifier = [];
end

%% Classify each tire individually
if ctrl.classifyEachTireIndividually
    % Train binary classifier

    featureNamesToBeDeletedRR = {'SPEED_FL','SPEED_FR','SPEED_RL',...
        'SVD',...
        'HEIGHT_FL','HEIGHT_FR','HEIGHT_RL',...
        'EXT_DOM_Z_VL', 'EXT_DOM_Z_VR', 'EXT_DOM_Z_HL',...
        'EXT_RT_X_VL', 'EXT_RT_X_VR', 'EXT_RT_X_HL',...
        'EXT_R_Z_VL', 'EXT_R_Z_VR', 'EXT_R_Z_HL'};
    labelMappingRR = {'RRDamperDefect', 'DamperDefect';...
        'allDampersDefect', 'DamperDefect';...
        'FLDamperDefect', 'passiveIntact';...
        'RRSpringPlus5_8PercStiffness', 'passiveIntact';...
        'RRSpringPlus16_4PercStiffness', 'passiveIntact';...
        'FLToeMinus17min', 'passiveIntact';...
        'FLToeMinus21min', 'passiveIntact';...
        'RearAxleDamperMinus20Percent', 'DamperDefect'};
    renameFeatureNamesMappingRR = {'SPEED_RR', 'SPEED';...
        'HEIGHT_RR', 'HEIGHT';...
        'EXT_DOM_Z_HR', 'EXT_DOM_Z';...
        'EXT_RT_X_HR', 'EXT_RT_X';...
        'EXT_R_Z_HR', 'EXT_R_Z'};
    [trainedClassifierRR, featuresTrainingRR] = trainBinaryClassifier(featuresTraining, featureNamesToBeDeletedRR, labelMappingRR, renameFeatureNamesMappingRR, classifierOpts, ctrl.trainClassifier);

    featureNamesToBeDeletedFL = {'SPEED_FR','SPEED_RL','SPEED_RR',...
    'SVD',...
    'HEIGHT_FR','HEIGHT_RL','HEIGHT_RR',...
    'EXT_DOM_Z_VR', 'EXT_DOM_Z_HL', 'EXT_DOM_Z_HR',...
    'EXT_RT_X_VR', 'EXT_RT_X_HL', 'EXT_RT_X_HR',...
    'EXT_R_Z_VR', 'EXT_R_Z_HL', 'EXT_R_Z_HR'};
    labelMappingFL = {'FLDamperDefect', 'DamperDefect';...
        'allDampersDefect', 'DamperDefect';...
        'RRDamperDefect', 'passiveIntact';...
        'RRSpringPlus5_8PercStiffness', 'passiveIntact';...
        'RRSpringPlus16_4PercStiffness', 'passiveIntact';...
        'FLToeMinus17min', 'passiveIntact';...
        'FLToeMinus21min', 'passiveIntact'};
    renameFeatureNamesMappingFL = {'SPEED_FL', 'SPEED';...
    'HEIGHT_FL', 'HEIGHT';...
    'EXT_DOM_Z_HL', 'EXT_DOM_Z';...
    'EXT_RT_X_HL', 'EXT_RT_X';...
    'EXT_R_Z_HL', 'EXT_R_Z'};
    [trainedClassifierFL, featuresTrainingFL] = trainBinaryClassifier(featuresTraining, featureNamesToBeDeletedFL, labelMappingFL, renameFeatureNamesMappingFL, classifierOpts, ctrl.trainClassifier);
end


%% Analyze Feature Importance Using Random Forest
if ctrl.analyzeFeatureImportanceUsingRandomForest
    featuresTraining.TreeImportance = featureImportanceUsingTree('featuresTraining',featuresTraining);
    if ctrl.classifyEachTireIndividually
        if exist('featuresTrainingRR', 'var')
            featuresTrainingRR.TreeImportance = featureImportanceUsingTree('featuresTraining',featuresTrainingRR);
        end
        if exist('featuresTrainingFL', 'var')
            featuresTrainingFL.TreeImportance = featureImportanceUsingTree('featuresTraining',featuresTrainingFL);
        end
    end
end

%% Performance on testing data
testdataTesting = testClassifier(trainedClassifier, featureExtractionHandle, opts, dataTesting);
if ctrl.trainClassifier
    testdataTesting.probabilityAnalysis = evaluateProbabilityPostProcessing(testdataTesting.features, featuresTraining, trainedClassifier);
end

if ctrl.classifyEachTireIndividually
    if exist('trainedClassifierRR', 'var')
        fprintf('\n\nPerformance on all RR testing data using the RR classifier\n');
        testdataTestingRR = testBinaryClassifier(trainedClassifierRR, testdataTesting, featureNamesToBeDeletedRR, labelMappingRR, renameFeatureNamesMappingRR, featureExtractionHandle, opts);
        if ~isempty(trainedClassifierRR)
            testdataTestingRR.probabilityAnalysis = evaluateProbabilityPostProcessing(testdataTestingRR.features, featuresTrainingRR, trainedClassifierRR);
        end
    end
    
    if exist('trainedClassifierFL','var')
        fprintf('\n\nPerformance on testing data using the FL classifier\n');
        testdataTestingFL = testBinaryClassifier(trainedClassifierFL, testdataTesting, featureNamesToBeDeletedFL, labelMappingFL, renameFeatureNamesMappingFL, featureExtractionHandle, opts);
        if ~isempty(trainedClassifierFL)
            testdataTestingFL.probabilityAnalysis = evaluateProbabilityPostProcessing(testdataTestingFL.features, featuresTrainingFL, trainedClassifierFL);
        end
    end
end

%% Performance on DD2 mass variation data set
filenameMass = 'Label_DD2_Massenvariation.xlsx';
foldernameMass = ['..', filesep, 'Datensatz', filesep];
fprintf('\n\n\n\n\n------------- Mass variation dataset -------------------\n');
testDD2Mass = testClassifier(trainedClassifier, featureExtractionHandle, opts, filenameMass, foldernameMass);
if ~isempty(trainedClassifier)
    testDD2Mass.probabilityAnalysis = evaluateProbabilityPostProcessing(testDD2Mass.features, featuresTraining, trainedClassifier);
end

if ctrl.classifyEachTireIndividually
    fprintf('\n\nPerformance on mass data using the FL classifier\n');
    testDD2MassFL = testBinaryClassifier(trainedClassifierFL, testDD2Mass, featureNamesToBeDeletedFL, labelMappingFL, renameFeatureNamesMappingFL, featureExtractionHandle, opts);
    if ~isempty(trainedClassifierFL)
        testDD2MassFL.probabilityAnalysis = evaluateProbabilityPostProcessing(testDD2MassFL.features, featuresTrainingFL, trainedClassifierFL);
    end
    fprintf('\n\nPerformance on mass data using the RR classifier\n');
    testDD2MassRR = testBinaryClassifier(trainedClassifierRR, testDD2Mass, featureNamesToBeDeletedRR, labelMappingRR, renameFeatureNamesMappingRR, featureExtractionHandle, opts);
    if ~isempty(trainedClassifierRR)
        testDD2MassRR.probabilityAnalysis = evaluateProbabilityPostProcessing(testDD2MassRR.features, featuresTrainingRR, trainedClassifierRR);
    end
end

%% Performance on DD2 tire variation data set
filenameTire = 'Label_DD_ReiterEngineering.xlsx';
foldernameTire = ['..', filesep, 'Datensatz', filesep];
fprintf('\n\n\n\n\n------------- Tire variation dataset -------------------\n');
testDD2Tire = testClassifier(trainedClassifier, featureExtractionHandle, opts, filenameTire, foldernameTire);
if ~isempty(trainedClassifier)
    testDD2Tire.probabilityAnalysis = evaluateProbabilityPostProcessing(testDD2Tire.features, featuresTraining, trainedClassifier);
end

if ctrl.classifyEachTireIndividually
    fprintf('Performance on Tire data using the FL classifier\n');
    testDD2TireFL = testBinaryClassifier(trainedClassifierFL, testDD2Tire, featureNamesToBeDeletedFL, labelMappingFL, renameFeatureNamesMappingFL, featureExtractionHandle, opts);
    if ~isempty(trainedClassifierFL)
        testDD2TireFL.probabilityAnalysis = evaluateProbabilityPostProcessing(testDD2TireFL.features, featuresTrainingFL, trainedClassifierFL);
    end
    fprintf('Performance on Tire data using the RR classifier\n');
    testDD2TireRR = testBinaryClassifier(trainedClassifierRR, testDD2Tire, featureNamesToBeDeletedRR, labelMappingRR, renameFeatureNamesMappingRR, featureExtractionHandle, opts);
    if ~isempty(trainedClassifierRR)
        testDD2TireRR.probabilityAnalysis = evaluateProbabilityPostProcessing(testDD2TireRR.features, featuresTrainingRR, trainedClassifierRR);
    end
end

%% Performance on rest of full data
if exist('unbalancedRestData', 'var')
    unbalancedRest = testClassifier(trainedClassifier, featureExtractionHandle, opts, unbalancedRestData);
    unbalancedRest.probabilityAnalysis = evaluateProbabilityPostProcessing(unbalancedRest.features, featuresTraining, trainedClassifier);
end

%% Save Workspace before RFE
if ctrl.saveData
    statusFolder = mkdir(ctrl.pathToSave);
    if statusFolder
        save(fullfile(ctrl.pathToSave, ctrl.matFileNameToSave));
    else
        % Create folder iteratively
        currPath = pwd;
        for pathPart = split(ctrl.pathToSave)
            if ~exist(pathPart{1},'dir')
                mkdir(pathPart{1});
            end
            cd(pathPart{1});
        end
        save(fullfile(ctrl.pathToSave, ctrl.matFileNameToSave));
        cd(currPath);
    end
end


%% Perform RFE (Recursive Feature Elimination) for Feature Analysis
if ctrl.performRFE
    %% RFE Using Accuracy
    if max(contains(ctrl.RFEType,'accuracy'))
        % Perform RFE using Accuracy (see function description for more details)
        if max(contains(ctrl.RFEType,'FeatureBlocks'))
            RFEusingAccuracyFeatBlocks = performRFEusingAccuracy(featuresTraining, 'evaluateFeatureBlocks', 1);
            
            [RFEusingAccuracyFeatBlocks.testdataTesting.accuracyUsingSelectedFeatures, RFEusingAccuracyFeatBlocks.testdataTesting.numberSelectedFeatures] = ...
            predictForEachFeatureSubset(testdataTesting.features, RFEusingAccuracyFeatBlocks.trainedClassifier);
            title('RFE FeatBlock Performance on test data using Accuracy and classifier trained on all training data');
            
            if exist('testDD2Mass','var')
                [RFEusingAccuracyFeatBlocks.testDD2Mass.accuracyUsingSelectedFeatures, RFEusingAccuracyFeatBlocks.testDD2Mass.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(testDD2Mass.features, RFEusingAccuracyFeatBlocks.trainedClassifier);
                title('RFE Performance using mass data');
            end
            if exist('testDD2Tire','var')
                [RFEusingAccuracyFeatBlocks.testDD2Tire.accuracyUsingSelectedFeatures, RFEusingAccuracyFeatBlocks.testDD2Tire.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(testDD2Tire.features, RFEusingAccuracyFeatBlocks.trainedClassifier);
                title('RFE Performance using tire data');
            end
            if exist('unbalancedRest','var')
                [RFEusingAccuracyFeatBlocks.unbalancedRest.accuracyUsingSelectedFeatures, RFEusingAccuracyFeatBlocks.unbalancedRest.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(unbalancedRest.features, RFEusingAccuracyFeatBlocks.trainedClassifier);
                title('RFE Performance using unbalanced rest of dataset');
            end
            
        end
        if max(contains(ctrl.RFEType,'SignalBlocks'))
            RFEusingAccuracySignalBlocks = performRFEusingAccuracy(featuresTraining, 'evaluateSignalBlocks', 1);
            
            [RFEusingAccuracySignalBlocks.testdataTesting.accuracyUsingSelectedFeatures, RFEusingAccuracySignalBlocks.testdataTesting.numberSelectedFeatures] = ...
            predictForEachFeatureSubset(testdataTesting.features, RFEusingAccuracySignalBlocks.trainedClassifier);
            title('RFE SignalBlock Performance on test data using Accuracy and classifier trained on all training data');
            
            if exist('testDD2Mass','var')
                [RFEusingAccuracySignalBlocks.testDD2Mass.accuracyUsingSelectedFeatures, RFEusingAccuracySignalBlocks.testDD2Mass.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(testDD2Mass.features, RFEusingAccuracySignalBlocks.trainedClassifier);
                title('RFE Performance using mass data');
            end
            if exist('testDD2Tire','var')
                [RFEusingAccuracySignalBlocks.testDD2Tire.accuracyUsingSelectedFeatures, RFEusingAccuracySignalBlocks.testDD2Tire.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(testDD2Tire.features, RFEusingAccuracySignalBlocks.trainedClassifier);
                title('RFE Performance using tire data');
            end
            if exist('unbalancedRest','var')
                [RFEusingAccuracySignalBlocks.unbalancedRest.accuracyUsingSelectedFeatures, RFEusingAccuracySignalBlocks.unbalancedRest.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(unbalancedRest.features, RFEusingAccuracySignalBlocks.trainedClassifier);
                title('RFE Performance using unbalanced rest of dataset');
            end
        end
        if max(strcmp(ctrl.RFEType,'accuracy'))
            RFEusingAccuracy = performRFEusingAccuracy(featuresTraining);
            
            [RFEusingAccuracy.testdataTesting.accuracyUsingSelectedFeatures, RFEusingAccuracy.testdataTesting.numberSelectedFeatures] = ...
            predictForEachFeatureSubset(testdataTesting.features, RFEusingAccuracy.trainedClassifier);
            title('RFE Performance on test data using Accuracy and classifier trained on all training data');
            
            if exist('testDD2Mass','var')
                [RFEusingAccuracy.testDD2Mass.accuracyUsingSelectedFeatures, RFEusingAccuracy.testDD2Mass.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(testDD2Mass.features, RFEusingAccuracy.trainedClassifier);
                title('RFE Performance using mass data');
            end
            if exist('testDD2Tire','var')
                [RFEusingAccuracy.testDD2Tire.accuracyUsingSelectedFeatures, RFEusingAccuracy.testDD2Tire.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(testDD2Tire.features, RFEusingAccuracy.trainedClassifier);
                title('RFE Performance using tire data');
            end
            if exist('unbalancedRest','var')
                [RFEusingAccuracy.unbalancedRest.accuracyUsingSelectedFeatures, RFEusingAccuracy.unbalancedRest.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(unbalancedRest.features, RFEusingAccuracy.trainedClassifier);
                title('RFE Performance using unbalanced rest of dataset');
            end
        end
        
        
        if exist('featuresTrainingRR','var')
            
            if max(contains(ctrl.RFEType,'FeatureBlocks'))
                RFEusingAccuracyFeatBlocksRR = performRFEusingAccuracy(featuresTrainingRR, 'evaluateFeatureBlocks', 1);
                
                % Performance of RFE selected features on all RR test data
                [RFEusingAccuracyFeatBlocksRR.testdataTestingRR.accuracyUsingSelectedFeatures, RFEusingAccuracyFeatBlocksRR.testdataTestingRR.numberSelectedFeatures] = ...
                predictForEachFeatureSubset(testdataTestingRR.features, RFEusingAccuracyFeatBlocksRR.trainedClassifier);
                title('RFE FeatBlock using accuracy on all RR test data trained on RR training data');
                
                if exist('testDD2MassRR','var')
                    [RFEusingAccuracyFeatBlocksRR.testDD2MassRR.accuracyUsingSelectedFeatures, RFEusingAccuracyFeatBlocksRR.testDD2MassRR.numberSelectedFeatures] = ...
                        predictForEachFeatureSubset(testDD2MassRR.features, RFEusingAccuracyFeatBlocksRR.trainedClassifier);
                    title('RFE Performance using mass data');
                end
                if exist('testDD2TireRR','var')
                    [RFEusingAccuracyFeatBlocksRR.testDD2TireRR.accuracyUsingSelectedFeatures, RFEusingAccuracyFeatBlocksRR.testDD2TireRR.numberSelectedFeatures] = ...
                        predictForEachFeatureSubset(testDD2TireRR.features, RFEusingAccuracyFeatBlocksRR.trainedClassifier);
                    title('RFE Performance using tire data');
                end
                
                % Performance of RFE selected features on RL test data
                if exist('testdataTestingRLusingRRclassifier', 'var')
                    [RFEusingAccuracyFeatBlocksRR.testdataTestingRLusingRRclassifier.accuracyUsingSelectedFeatures, RFEusingAccuracyFeatBlocksRR.testdataTestingRLusingRRclassifier.numberSelectedFeatures] = ...
                        predictForEachFeatureSubset(testdataTestingRLusingRRclassifier.features, RFEusingAccuracyFeatBlocksRR.trainedClassifier);
                    title('RFE FeatBlock using accuracy on RL test data trained on RR training data');
                end
            end
            
            if max(contains(ctrl.RFEType,'SignalBlocks'))
                RFEusingAccuracySignalBlocksRR = performRFEusingAccuracy(featuresTrainingRR, 'evaluateSignalBlocks', 1);
                
                % Performance of RFE selected features on all RR test data
                [RFEusingAccuracySignalBlocksRR.testdataTestingRR.accuracyUsingSelectedFeatures, RFEusingAccuracySignalBlocksRR.testdataTestingRR.numberSelectedFeatures] = ...
                predictForEachFeatureSubset(testdataTestingRR.features, RFEusingAccuracySignalBlocksRR.trainedClassifier);
                title('RFE SignalBlock using accuracy on all RR test data trained on RR training data');
                
                if exist('testDD2MassRR','var')
                    [RFEusingAccuracySignalBlocksRR.testDD2MassRR.accuracyUsingSelectedFeatures, RFEusingAccuracySignalBlocksRR.testDD2MassRR.numberSelectedFeatures] = ...
                        predictForEachFeatureSubset(testDD2MassRR.features, RFEusingAccuracySignalBlocksRR.trainedClassifier);
                    title('RFE Performance using mass data');
                end
                if exist('testDD2TireRR','var')
                    [RFEusingAccuracySignalBlocksRR.testDD2TireRR.accuracyUsingSelectedFeatures, RFEusingAccuracySignalBlocksRR.testDD2TireRR.numberSelectedFeatures] = ...
                        predictForEachFeatureSubset(testDD2TireRR.features, RFEusingAccuracySignalBlocksRR.trainedClassifier);
                    title('RFE Performance using tire data');
                end
                
                % Performance of RFE selected features on RL test data
                if exist('testdataTestingRLusingRRclassifier', 'var')
                    [RFEusingAccuracySignalBlocksRR.testdataTestingRLusingRRclassifier.accuracyUsingSelectedFeatures, RFEusingAccuracySignalBlocksRR.testdataTestingRLusingRRclassifier.numberSelectedFeatures] = ...
                        predictForEachFeatureSubset(testdataTestingRLusingRRclassifier.features, RFEusingAccuracySignalBlocksRR.trainedClassifier);
                    title('RFE SignalBlock using accuracy on RL test data trained on RR training data');
                end
            end
            
            if max(strcmp(ctrl.RFEType,'accuracy'))
                RFEusingAccuracyRR = performRFEusingAccuracy(featuresTrainingRR);
                
                % Performance of RFE selected features on all RR test data
                [RFEusingAccuracyRR.testdataTestingRR.accuracyUsingSelectedFeatures, RFEusingAccuracyRR.testdataTestingRR.numberSelectedFeatures] = ...
                predictForEachFeatureSubset(testdataTestingRR.features, RFEusingAccuracyRR.trainedClassifier);
                title('RFE using accuracy on all RR test data trained on RR training data');
                
                if exist('testDD2MassRR','var')
                    [RFEusingAccuracyRR.testDD2MassRR.accuracyUsingSelectedFeatures, RFEusingAccuracyRR.testDD2MassRR.numberSelectedFeatures] = ...
                        predictForEachFeatureSubset(testDD2MassRR.features, RFEusingAccuracyRR.trainedClassifier);
                    title('RFE Performance using mass data');
                end
                if exist('testDD2TireRR','var')
                    [RFEusingAccuracyRR.testDD2TireRR.accuracyUsingSelectedFeatures, RFEusingAccuracyRR.testDD2TireRR.numberSelectedFeatures] = ...
                        predictForEachFeatureSubset(testDD2TireRR.features, RFEusingAccuracyRR.trainedClassifier);
                    title('RFE Performance using tire data');
                end
                
                % Performance of RFE selected features on RL test data
                if exist('testdataTestingRLusingRRclassifier', 'var')
                    [RFEusingAccuracyRR.testdataTestingRLusingRRclassifier.accuracyUsingSelectedFeatures, RFEusingAccuracyRR.testdataTestingRLusingRRclassifier.numberSelectedFeatures] = ...
                        predictForEachFeatureSubset(testdataTestingRLusingRRclassifier.features, RFEusingAccuracyRR.trainedClassifier);
                    title('RFE using accuracy on RL test data trained on RR training data');
                end
            end
            
        end
        
    end
    
    %% Save Workspace before RFE using weights
    if ctrl.saveData
        statusFolder = mkdir(ctrl.pathToSave);
        if statusFolder
            save(fullfile(ctrl.pathToSave, ctrl.matFileNameToSave));
        else
            % Create folder iteratively
            currPath = pwd;
            for pathPart = split(ctrl.pathToSave)
                if ~exist(pathPart{1},'dir')
                    mkdir(pathPart{1});
                end
                cd(pathPart{1});
            end
            save(fullfile(ctrl.pathToSave, ctrl.matFileNameToSave));
            cd(currPath);
        end
    end
    
    %% RFE using weights
    if max(contains(ctrl.RFEType,'weights'))
        % Perform RFE using weights of a linear SVM (see function description for more details)
        RFEusingWeights = performRFEusingWeights(featuresTraining, opts);
        
        % Performance of RFE selected features
        [RFEusingWeights.testdataTesting.accuracyUsingSelectedFeatures, RFEusingWeights.testdataTesting.numberSelectedFeatures] = ...
            predictForEachFeatureSubset(testdataTesting.features, RFEusingWeights.trainedClassifier);
        title('RFE Performance using weights on test data and classifier trained on all training data');
        
        if exist('featuresTrainingRR','var')
            % Perform RFE using weights of a linear SVM (see function description for more details)
            RFEusingWeightsRR = performRFEusingWeights(featuresTrainingRR, opts);

            % Performance of RFE selected features
            [RFEusingWeightsRR.testdataTestingRR.accuracyUsingSelectedFeatures, RFEusingWeightsRR.testdataTestingRR.numberSelectedFeatures] = ...
                predictForEachFeatureSubset(testdataTestingRR.features, RFEusingWeightsRR.trainedClassifier);
            title('RFE Performance using weights on test data and classifier trained on RR data');
            
            if exist('testdataTestingRLusingRRclassifier', 'var')
                % Performance of RFE selected features on RL test data
                [RFEusingWeightsRR.testdataTestingRLusingRRclassifier.accuracyUsingSelectedFeatures, RFEusingWeightsRR.testdataTestingRLusingRRclassifier.numberSelectedFeatures] = ...
                    predictForEachFeatureSubset(testdataTestingRLusingRRclassifier.features, RFEusingWeightsRR.trainedClassifier);
                title('RFE Performance using weights on RL test data and classifier trained on RR training data');
            end
        end
        
        if exist('testDD2Mass','var')
            [RFEusingWeights.testDD2Mass.accuracyUsingSelectedFeatures, RFEusingWeights.testDD2Mass.numberSelectedFeatures] = ...
                predictForEachFeatureSubset(testDD2Mass.features, RFEusingWeights.trainedClassifier);
            title('RFE Performance using mass data');
        end
        if exist('testDD2Tire','var')
            [RFEusingWeights.testDD2Tire.accuracyUsingSelectedFeatures, RFEusingWeights.testDD2Tire.numberSelectedFeatures] = ...
                predictForEachFeatureSubset(testDD2Tire.features, RFEusingWeights.trainedClassifier);
            title('RFE Performance using tire data');
        end
        if exist('unbalancedRest','var')
            [RFEusingWeights.unbalancedRest.accuracyUsingSelectedFeatures, RFEusingWeights.unbalancedRest.numberSelectedFeatures] = ...
                predictForEachFeatureSubset(unbalancedRest.features, RFEusingWeights.trainedClassifier);
            title('RFE Performance using unbalanced rest of dataset');
        end
    end
end

%% Save Workspace after RFE
if ctrl.saveData
    statusFolder = mkdir(ctrl.pathToSave);
    if statusFolder
        save(fullfile(ctrl.pathToSave, ctrl.matFileNameToSave));
    else
        % Create folder iteratively
        currPath = pwd;
        for pathPart = split(ctrl.pathToSave)
            if ~exist(pathPart{1},'dir')
                mkdir(pathPart{1});
            end
            cd(pathPart{1});
        end
        save(fullfile(ctrl.pathToSave, ctrl.matFileNameToSave));
        cd(currPath);
    end
end

%% Data Size Analysis
if ctrl.performDataSizeAnalysis
    if strcmp(ctrl.featureType, 'Autoencoder')
        datasizeAnalysis = analyzePerformanceWithReducedTrainingDataRepLearn(dataTraining, opts, ctrl, 'classifierOpts', classifierOpts, 'autoencoderOpts', autoencoderOpts);
    elseif strcmp(ctrl.featureType, 'Sparsefilter')
        datasizeAnalysis = analyzePerformanceWithReducedTrainingDataRepLearn(dataTraining, opts, ctrl, 'classifierOpts', classifierOpts, 'sparsefilterOpts', sparsefilterOpts);
    else
        datasizeAnalysis = analyzePerformanceWithReducedTrainingDataRepLearn(dataTraining, opts, ctrl, 'classifierOpts', classifierOpts, 'featureExtractionHandle', featureExtractionHandle);
    end
    
    [datasizeAnalysis.testdataTesting.accuracyUsingSelectedFeatures] = ...
        predictForEachDatasize(dataTesting, datasizeAnalysis, 'plotOutput',0);
    
    if exist('testDD2Mass','var')
        [datasizeAnalysis.testDD2Mass.accuracyUsingSelectedFeatures] = ...
            predictForEachDatasize(testDD2Mass.data, datasizeAnalysis, 'plotOutput',0);
    end
    if exist('testDD2Tire','var')
        [datasizeAnalysis.testDD2Tire.accuracyUsingSelectedFeatures] = ...
            predictForEachDatasize(testDD2Tire.data, datasizeAnalysis, 'plotOutput',0);
    end
    
    if exist('featuresTrainingRR','var')
        datasizeAnalysisRR = analyzePerformanceWithReducedTrainingData(featuresTrainingRR, 'classifierOpts', classifierOpts);
        
        [datasizeAnalysisRR.testdataTestingRR.accuracyUsingSelectedFeatures] = ...
            predictForEachFeatureSubset(testdataTestingRR.features, datasizeAnalysisRR.trainedClassifier, 'plotOutput',0);
        
        if exist('testdataTestingRLusingRRclassifier', 'var')
            [datasizeAnalysisRR.testdataTestingRLusingRRclassifier.accuracyUsingSelectedFeatures] = ...
                predictForEachFeatureSubset(testdataTestingRLusingRRclassifier.features, datasizeAnalysisRR.trainedClassifier, 'plotOutput',0);
        end
        if exist('testDD2MassRR','var')
            [datasizeAnalysisRR.testDD2MassRR.accuracyUsingSelectedFeatures] = ...
                predictForEachFeatureSubset(testDD2MassRR.features, datasizeAnalysisRR.trainedClassifier, 'plotOutput',0);
        end
        if exist('testDD2TireRR','var')
            [datasizeAnalysisRR.testDD2TireRR.accuracyUsingSelectedFeatures] = ...
                predictForEachFeatureSubset(testDD2TireRR.features, datasizeAnalysisRR.trainedClassifier, 'plotOutput',0);
        end
    end
end

%% Save Workspace
if ctrl.saveData
    statusFolder = mkdir(ctrl.pathToSave);
    if statusFolder
        save(fullfile(ctrl.pathToSave, ctrl.matFileNameToSave));
    else
        % Create folder iteratively
        currPath = pwd;
        for pathPart = split(ctrl.pathToSave)
            if ~exist(pathPart{1},'dir')
                mkdir(pathPart{1});
            end
            cd(pathPart{1});
        end
        save(fullfile(ctrl.pathToSave, ctrl.matFileNameToSave));
        cd(currPath);
    end
    
    % Save command window and set diary-mode off
    diary off
end