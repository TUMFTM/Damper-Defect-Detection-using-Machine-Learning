
addpath(genpath('W:\Projekte\Fahrwerkdiagnose\Datenanalyse\Skripte'));

filename = 'Label_DD2_Massenvariation.xlsx';
% filename = 'Label_DD.xlsx';
foldername = 'W:\Projekte\Fahrwerkdiagnose\Datenanalyse\Datensatz\';

%% opts = setOptions();
opts.data = struct('ACC_X',[],'ACC_Y',[],'YAW_RATE',[],...
    'SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[],...
    'Prop',[]);
% Flexray
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
opts.fs = 100;           % Sampling frequency [Hz]
% opts.t = 5;             % Analysis block time [s] resulting from number of data points
opts.t = 512/opts.fs;    % Analysis block time [s] resulting from number of data points
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

opts.detrend = 0;
opts.detrend_mode = 'movmean'; % 'const', 'linear', 'standardize', 'movmean'

%%
opts.filename = filename;
opts.foldername = foldername;

% Load data
[data, tf] = loadData(filename, foldername, opts);

analyseDataset(data, opts)