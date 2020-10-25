% clear;clc;
tic;
addpath(genpath('C:\Users\tzehe\Documents\Fahrwerkdiagnose\Datenanalyse\Scripts'));

% uncomment the following command to load data
% load('C:\Users\tzehe\Documents\Fahrwerkdiagnose\Datenanalyse\BMW\Jautze\200817_2040_optim_to_ROC_w_dataTraining_complete\Workspace.mat')

%% Skriptsteuerung
ctrl.loadMatFile = 1;
ctrl.loadMatFilePath = "C:\Users\tzehe\Documents\Fahrwerkdiagnose\Datenanalyse\BMW\MachineLearning\FFT\200401_1023\Workspace.mat";
ctrl.loadData = 0;
ctrl.saveData = 1;
ctrl.pathToSave = ['C:\Users\tzehe\Documents\Fahrwerkdiagnose\Datenanalyse\BMW\Jautze\', datestr(now, 'yymmdd_HHMM'), '\'];
ctrl.matFileNameToSave = 'Workspace.mat';
ctrl.loadOptimizationResult = 0;
ctrl.pathOptimizationResult = 'C:\Users\tzehe\Documents\Fahrwerkdiagnose\Datenanalyse\BMW\Jautze\200816_1841_optim_dataTraining_complete\Workspace.mat';

ctrl.plotHistoFFT = 1;
ctrl.plotQuantile = 0.99;     % end of histogram x-axis
ctrl.nHistoBars = 50;      % number of histogram bars
ctrl.plot_AUC = 1;
ctrl.plot_Pxx = 0;
ctrl.optimize_for_AUC = 1;
ctrl.corr_factor_nbins = 15;

ctrl.neighboring_fft_points = 0;    % number of neighboring fft datapoints for fft-based DSKW calculation -> overall number of datapoints = 2 * neighboring_datapoints + 1
ctrl.set_illconditioned_DSKW_as_Nan = false;
ctrl.eliminate_mean = true;

%% Set some parameters
fAus = 15;
fRef = 24;
fRef2 = fRef.^2./fAus;
f_vec = [fAus, fRef];
% fRef2(fRef2>fs/2) = 25;
% wähle die Mittelwerte von fAus, fRef und fRef2 so dass gilt
% fAus/fRef ist ungefähr fRef/fRef2


%% Load data
if ctrl.loadData
    filename = 'Label_DD2.xlsx';
    foldername = ['..', filesep, 'Datensatz', filesep];
    opts = setOptions('dataSource','Flexray');
    data = loadData(filename, foldername, opts);
elseif ctrl.loadMatFile
    % load mat file and restore ctrl-variable
    clearvars -except ctrl
    ctrl_backup = ctrl;
    load(ctrl.loadMatFilePath);
    ctrl_backup_fieldnames = fieldnames(ctrl_backup);
    for cnt_field = 1 : length(ctrl_backup_fieldnames)
        ctrl.(ctrl_backup_fieldnames{cnt_field}) = ctrl_backup.(ctrl_backup_fieldnames{cnt_field});
    end
    clear ctrl_backup ctrl_backup_fieldnames
end

%% Create folder for saving
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

%% select only observations with special properties (e.g. speed > 80 km/h)
% data_for_analysis = reduceDataWithProperty(dataTraining, 'meanWheelspeed', @(x)x>(90/(opts.d_wheel/2*3.6)));
% data_for_analysis = reduceDataWithProperty(data_for_analysis, 'meanWheelspeed', @(x)x<(110/(opts.d_wheel/2*3.6)));
% data_for_analysis = reduceDataWithProperty(data_for_analysis, 'meanAx', @(x)abs(x)<0.1);
% data_for_analysis = reduceDataWithProperty(data_for_analysis, 'meanAy', @(x)abs(x)<0.1);
data_for_analysis = dataTraining;
data_for_analysis = detrendData(data_for_analysis, opts);
fprintf('using %d observations of dataTraining as data_for_analysis\n', size(data_for_analysis.SPEED_FL,1));

%% Convert Labels
is_defect = convert_labels(data_for_analysis);

%% Calculate DSKW
DSKW = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
DSKW_Ref2 = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
Pxx = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
AUC = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
fpr = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
tpr = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
speedNames = fieldnames(DSKW);
if ctrl.plot_Pxx
    hPxx = figure('Name','Pxx_optim');
else
    hPxx = 0;
end
if ctrl.plot_AUC
    hROC = figure('Name','ROC_optim');
    hROC_Ref2 = figure('Name','ROC_Ref2_optim');
else
    hROC = 0;
    hROC_Ref2 = 0;
end

if ctrl.optimize_for_AUC
    result_optim = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
end
for cntSpeed = 1 : length(speedNames)

    speedName = speedNames{cntSpeed};
    speedForAnalysis = data_for_analysis.(speedName)';
    
    if ctrl.optimize_for_AUC
        if ~ctrl.loadOptimizationResult
            fun = @(x)getAUC_of_perform_Jautze_for_signal(data_for_analysis.(speedName)', x, is_defect.(speedName), ctrl, opts, 0, 0, cntSpeed, speedName);
            options = gaoptimset('Display','iter', 'UseParallel', 1, 'StallGenLimit', 10);
            result_optim.(speedName) = ga(fun,2,[],[],[],[],[10; 20],[17; 30],[],options);
            f_vec = result_optim.(speedName);
            speedName
            result_optim.(speedName)
        else
            fprintf('loading optimization result from %s \n', ctrl.pathOptimizationResult);
            load(ctrl.pathOptimizationResult, 'result_optim');
            f_vec = result_optim.(speedName);
            result_optim
        end
    end
    
    [DSKW.(speedName), DSKW_Ref2.(speedName), AUC.(speedName), AUC_Ref2.(speedName), Pxx.(speedName)] = perform_Jautze_for_signal(data_for_analysis.(speedName)', f_vec, is_defect.(speedName), ctrl, opts, hROC, hROC_Ref2, cntSpeed, speedName);
    
    %% Plot Pxx
    if ctrl.plot_Pxx
        figure(hPxx);
        subplot(2,2,cntSpeed);
        f = linspace(0,opts.fs/2, size(Pxx.(speedName),1));
        classes = unique(data_for_analysis.Prop.labelIsolation);
        for cntClass = 1 : size(classes,1)
            idx_for_mean = strcmp(data_for_analysis.Prop.labelIsolation,classes{cntClass});
            Pxx_mean = mean(Pxx.(speedName)(:,idx_for_mean),2);
            plot(f, Pxx_mean, 'DisplayName', classes{cntClass});
            hold on;
        end
        title(speedName,'Interpreter','None');
        xlabel('Frequency in Hz');
        ylabel('Pxx');
        legend('Location','northeast','Interpreter','None');
        grid on
    end
end

if ctrl.optimize_for_AUC
    DSKW_optim = DSKW;
    DSKW_Ref2_optim = DSKW_Ref2;
    AUC_optim = AUC;
    AUC_Ref2_optim = AUC_Ref2;
    Pxx_optim = Pxx;
end

%% Get gradient of DSKW with respect to vehicle speed and correct DSKW
dataTraining_detrended = detrendData(dataTraining, opts);

% Convert Labels
is_defect = convert_labels(dataTraining_detrended);
if ctrl.plot_AUC
    hROC = figure('Name','ROC_Training');
    hROC_Ref2 = figure('Name','ROC_Ref2_Training');
    hROC_corr = figure('Name','ROC_Training_corrected_by_speed');
else
    hROC = 0;
    hROC_Ref2 = 0;
    hROC_corr = 0;
end

hScatter = figure('Name', 'DSKW_vs_Speed_Training');
DSKW_corr_factor = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
DSKW_corrected_by_speed = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
AUC_corrected_by_speed = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
DSKW = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
DSKW_Ref2 = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
Pxx = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
AUC = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
fpr = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
tpr = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
for cntSpeed = 1 : length(speedNames)
    
    speedName = speedNames{cntSpeed};
%     speedForAnalysis = data_for_analysis.(speedName)';
    
    f_vec = result_optim.(speedName);
    [DSKW.(speedName), DSKW_Ref2.(speedName), AUC.(speedName), AUC_Ref2.(speedName), Pxx.(speedName)] = perform_Jautze_for_signal(dataTraining_detrended.(speedName)', f_vec, is_defect.(speedName), ctrl, opts, hROC, hROC_Ref2, cntSpeed, speedName);

    % calculate correction factor for DSKW (normalize to speed=100 rad/s)
    DSKW_corr_factor.(speedName) = struct('Speed',[],'corr_factor',[]);
    [~, edges, bins] = histcounts(dataTraining_detrended.Prop.meanWheelspeed(is_defect.(speedName)==0),ctrl.corr_factor_nbins);
    binned_median = splitapply(@median, DSKW.(speedName)(is_defect.(speedName)==0), bins);
    bin_center = edges(1:end-1) + diff(edges)/2;
    DSKW_corr_factor.(speedName).Speed = bin_center;
    idx_corr_bin_center = find(bin_center >= 100/(opts.d_wheel/2*3.6), 1, 'first');
    DSKW_corr_factor.(speedName).corr_factor = binned_median(idx_corr_bin_center) ./ binned_median;
    
    figure(hScatter);
    subplot(2,2,cntSpeed);
    [~, edges, bins] = histcounts(dataTraining_detrended.Prop.meanWheelspeed(is_defect.(speedName)==1),ctrl.corr_factor_nbins);
    binned_median = splitapply(@median, DSKW.(speedName)(is_defect.(speedName)==1), bins);
    bin_center = edges(1:end-1) + diff(edges)/2;
    plot(bin_center*opts.d_wheel/2*3.6, binned_median,'DisplayName', 'defect');
    hold on
    
    [~, edges, bins] = histcounts(dataTraining_detrended.Prop.meanWheelspeed(is_defect.(speedName)==0),ctrl.corr_factor_nbins);
    binned_median = splitapply(@median, DSKW.(speedName)(is_defect.(speedName)==0), bins);
    bin_center = edges(1:end-1) + diff(edges)/2;
    plot(bin_center*opts.d_wheel/2*3.6, binned_median,'DisplayName', 'intact');
    legend show
    xlabel('Speed in km/h');
    ylabel('DSKW');
    
    DSKW_corrected_by_speed.(speedName) = DSKW.(speedName).*interp1(DSKW_corr_factor.(speedName).Speed, DSKW_corr_factor.(speedName).corr_factor, dataTraining_detrended.Prop.meanWheelspeed);
    
    % Calculate corrected DSKW values AUC values
    if ctrl.plot_AUC
        figure(hROC_corr);
        subplot(2,2,cntSpeed);
    end
    [AUC_corrected_by_speed.(speedName), ~, ~] = fastAUC(logical(is_defect.(speedName)),DSKW_corrected_by_speed.(speedName),ctrl.plot_AUC);
    if ctrl.plot_AUC
        title([speedName, ' AUC = ', num2str(AUC_corrected_by_speed.(speedName))],'Interpreter','None');
    end
end

%% Calc corrected DSKW for dataTesting
[DSKW_corrected_by_speed_testing, AUC_corrected_by_speed_testing, DSKW_testing, DSKW_Ref2_testing, Pxx_testing] = testJautze(dataTesting, opts, ctrl, result_optim, DSKW_corr_factor);
[DSKW_corrected_by_speed_mass, AUC_corrected_by_speed_mass, DSKW_mass, DSKW_Ref2_mass, Pxx_mass] = testJautze(testDD2Mass.data, opts, ctrl, result_optim, DSKW_corr_factor);
[DSKW_corrected_by_speed_tire, AUC_corrected_by_speed_tire, DSKW_tire, DSKW_Ref2_tire, Pxx_tire] = testJautze(testDD2Tire.data, opts, ctrl, result_optim, DSKW_corr_factor);

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
