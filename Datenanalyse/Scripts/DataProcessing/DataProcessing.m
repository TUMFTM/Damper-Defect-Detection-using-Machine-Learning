%% Initialize
clear all
close all
clc
%% CTRL
ctrl.loadData = 1;
ctrl.calcFeatures = 1;  % 0 = no calculation of Features, 1 = calculate features
ctrl.reduceFeatures = 0;    % 0 = no reduction of features, 1 = reduce features
ctrl.export = 0;
ticDataProcessing = tic;

addpath(genpath('W:\Projekte\Fahrwerkdiagnose\Datenanalyse\Skripte'));

%% Parameters
% opts.data = struct('ACC_X',[],'ACC_Y',[],'YAW_RATE',[],...
%     'SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[],...
%     'STEERING_ANGLE',[],...
%     'Prop',[]);
opts.data = struct('ACC_X',[],'ACC_Y',[],'YAW_RATE',[],...
    'SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[],...
    'Prop',[]);
% opts.data = struct('ACC_X',[],'ACC_Y',[],'YAW_RATE',[],'SIDESLIP_ANGLE',[],...
%     'SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[],...
%     'STEERING_WHEEL_ANGLE',[],'STEERING_ANGLE',[],...
%     'Prop',[]);
% opts.data = struct('ACC_X',[],'ACC_Y',[],'SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[],'Prop',[]);

% opts.sensor_names = {possible names of sensor signal, desired name for scripts
% this variable can have more entries than actually needed for opts.data
% opts.sensor_names = {'ACLNX_COG','ACC_X';...
%         'ACLNY_COG','ACC_Y';...
%         'VYAW_VEH','YAW_RATE';...
%         'AVL_RPM_WHL_FLH','SPEED_FL';...
%         'AVL_RPM_WHL_FRH','SPEED_FR';...
%         'AVL_RPM_WHL_RLH','SPEED_RL';...
%         'AVL_RPM_WHL_RRH','SPEED_RR';...
%         'AVL_STEA_DV','STEERING_WHEEL_ANGLE';...
%         'STEA_FTAX_EFFV','STEERING_ANGLE';...
%         'ATTA_ESTI','SIDESLIP_ANGLE'};
opts.sensor_names = {'A_FlexRay__ACLNX_MASSCNTR__ACLNX_COG','ACC_X';...
            'A_FlexRay__ACLNY_MASSCNTR__ACLNY_COG','ACC_Y';...
            'A_FlexRay__VYAW_VEH__VYAW_VEH','YAW_RATE';...
            'A_FlexRay__AVL_RPM_WHL__AVL_RPM_WHL_FLH','SPEED_FL';...
            'A_FlexRay__AVL_RPM_WHL__AVL_RPM_WHL_FRH','SPEED_FR';...
            'A_FlexRay__AVL_RPM_WHL__AVL_RPM_WHL_RLH','SPEED_RL';...
            'A_FlexRay__AVL_RPM_WHL__AVL_RPM_WHL_RRH','SPEED_RR';...
            'AVL_STEA_DV','STEERING_WHEEL_ANGLE';...
            'STEA_FTAX_EFFV','STEERING_ANGLE';...
            'ATTA_ESTI','SIDESLIP_ANGLE'};

opts.useParallel = 1;   % 0 = run serial, 1 = run parallel
% opts.fs = 50;           % Sampling frequency [Hz]
opts.fs = 100;           % Sampling frequency [Hz]
% opts.t = 5;             % Analysis block time [s] resulting from number of data points
opts.t = 256/opts.fs;    % Analysis block time [s] resulting from number of data points
opts.overlap = 0;       % let windows overlap by given amount of datapoints for data augmentation
opts.min_speed = 30;     % Speed threshold [km/h]
opts.max_acc_x = 1;      % Acceleration_x threshold [m/s^2]
opts.max_acc_y = 1;      % Acceleration_y threshold [m/s^2]
opts.useLabel = 'labelIsolation';   % specify name of used label according to Excel table
opts.d_wheel = 0.635;        % wird für Umrechung Raddrehzahl -> Geschwindigkeit benötigt
opts.splitTrainingData = 0.7;        % percentage of data kept for training, rest for test

% identify default number of parallel workers
defaultProfile = parallel.defaultClusterProfile;
parallelCluster = parcluster(defaultProfile);
opts.numParallelWorkers = parallelCluster.NumWorkers;

opts.fieldsData = fields(opts.data);
opts.fieldsData = opts.fieldsData(~contains(opts.fieldsData,'Prop'));   % remove 'Prop' from fieldsData variable
opts.n_sensors = length(opts.fieldsData);

opts.detrend = 1;
opts.detrend_mode = 'movmean'; % 'const', 'linear', 'standardize'´, 'movmean'

opts = setOptions();

if ctrl.loadData == 1
    %% Load, resample, cut & merge data
    [filename, folder_name] = uigetfile('*.xlsx;*.xls','Please select Excel-File measurement overview','MultiSelect','on');


    %% Get labeling data
    if iscell(filename)     % if-cause and for-loop is needed for MultiSelect
        Label_Data = table;
        for cntLabelFiles = 1 : length(filename)
            Label_Data = [Label_Data; readtable([folder_name, filename{cntLabelFiles}])];
        end
    else
        Label_Data = readtable([folder_name, filename]);
    end
    fprintf('Load labeling data successful.\n');


    %% Load Data
    data = inputMeasurement(Label_Data,opts);
    
    %% Divide data
    n = size(data.(opts.fieldsData{1}),1);
    tf = false(n,1);
    tf(1:round(opts.splitTrainingData*n)) = true;
    tf = tf(randperm(n));
end


%% Detrend data
if ctrl.detrend == 1
    data = detrendData(data, opts);
else
    disp('Skipping detrend of data');
end

%% Feature Extraction
if ctrl.calcFeatures
    features = featureExtraction(data,opts);
    DATA_classification = features2table(features,opts);
    DATA_classification = rearrangeFeatureTable(DATA_classification);
end

%% Reduce Features
if ctrl.reduceFeatures
    if ctrl.calcFeatures == 0
        load('W:\Projekte\Fahrwerkdiagnose\Datenanalyse\Datensatz\DD\DD_allFeat_10Sens_RFE_finalSVMAccu_reducedSize.mat');
    end
    % load ranking of features
    selectedFeatures = load('W:\Projekte\Fahrwerkdiagnose\Datenanalyse\Datensatz\DD\DD_allFeat_10Sens_RFE_finalSVMAccu_reducedSize.mat','FeatureSelectionRFE');
    numSelectedFeatures = 100;
    DATA_classification_red = reduceFeatures(DATA_classification,selectedFeatures.FeatureSelectionRFE.rank_blocks(1:numSelectedFeatures));
    DATA_classificationSingFeat = rearrangeFeatureTable(DATA_classification);
    DATA_classification_singFeat_red = reduceFeatures(DATA_classificationSingFeat,selectedFeatures.FeatureSelectionRFE.rank_blocks(1:numSelectedFeatures));

end


%% Divide data
if ctrl.calcFeatures
    dataTraining = DATA_classification(tf,:);
    dataTesting = DATA_classification(~tf,:);
    
%     dataTrainingSingFeat = DATA_classificationSingFeat(tf,:);
%     dataTestingSingFeat = DATA_classificationSingFeat(~tf,:);
%     
%     dataTraining_redFeatBlocks = DATA_classification_redFeatBlocks(tf,:);
%     dataTesting_redFeatBlocks = DATA_classification_redFeatBlocks(~tf,:);
end


%% Export data
if ctrl.export
    data = exportData(data, opts, tf, ctrl);
end

%% Elapsed time
elapsedTime = toc(ticDataProcessing);
if ctrl.calcFeatures
    fprintf('Elapsed time: %.2f s (%.3f s per observation).\n',elapsedTime,elapsedTime/n);
    if iscell(filename)     % if-cause and for-loop is needed for MultiSelect
        for cntLabelFiles = 1 : length(filename)
            fprintf('Processed data from %s \n', [folder_name, filename{cntLabelFiles}]);
        end
    else
        fprintf('Processed data from %s \n', [folder_name, filename]);
    end
else
    fprintf('Elapsed time: %.2f sec\n',elapsedTime);
end

