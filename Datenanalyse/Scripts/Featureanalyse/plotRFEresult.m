clear;clc;

folder_vec{1} = '.\..\BMW\MachineLearning\ManuelleFeatures\200401_1025\Workspace.mat';
folder_vec{2} = '.\..\BMW\MachineLearning\FFT\200401_1023\Workspace.mat';
folder_vec{3} = '.\..\BMW\RepresentationLearning\Autoencoder\200402_1114\Workspace.mat';
folder_vec{4} = '.\..\BMW\RepresentationLearning\Sparsefilter\200402_1838\Workspace.mat';
save_path_comparison = '.\..\BMW\SupervisedComparison_RFE\';

save_plot = 1;
plot_individual_dataset = 1;
smooth_span_single_features = 5;    % parameter for smoothing the visualization of the RFE for single features, 5 is default, 1 is no smoothing, should be an odd value

if save_plot
    % Activate logging of Command Window
    diary(fullfile(save_path_comparison, ['CommandWindowLog.txt']));
end

varNames = {'featureBlocks','RFEusingAccuracyFeatBlocks';
    'signalBlocks','RFEusingAccuracySignalBlocks';
    'singleFeatures','RFEusingAccuracy'};

% varNames = {'signalBlocks','RFEusingAccuracySignalBlocks';
%     'singleFeatures','RFEusingWeights'};

for cntFolder = 1 : length(folder_vec)
    
    folder = folder_vec{cntFolder};

    folder_split = split(folder,'\');
    folder_short = folder_split{end-2};
    
    RFE = struct();
    for cntVar = 1 : size(varNames,1)
        try
            data = load(folder,varNames{cntVar,2});
            data.(varNames{cntVar,2}) = rmfield(data.(varNames{cntVar,2}), 'trainedClassifier');  % save memory
            RFE.(varNames{cntVar,1}) = data.(varNames{cntVar,2});
        end
    end

    fieldsRFE = fieldnames(RFE);

    if ~exist('hAll', 'var')
        for cntRFEtype = 1 : length(fieldsRFE)
            hAll(cntRFEtype) = figure();
            hold on
        end
    end
    
    for cntRFEtype = 1 : length(fieldsRFE)
        if plot_individual_dataset
            h(1) = figure;
            hold on;
        end
        RFEtype = fieldsRFE{cntRFEtype};
        datasetNames = fieldnames(RFE.(RFEtype));
        datasetNames = datasetNames(contains(datasetNames,["Testing","Mass","Tire"]));
        for cntDataset = 1 : length(datasetNames)
            datasetName = datasetNames{cntDataset};
            if plot_individual_dataset
                figure(h(1));
                if strcmp(RFEtype,'singleFeatures')
                    plot(RFE.(RFEtype).(datasetName).numberSelectedFeatures, RFE.(RFEtype).(datasetName).accuracyUsingSelectedFeatures.mean,'LineWidth',1,'DisplayName',datasetName);
                else
                    errorbar(RFE.(RFEtype).(datasetName).numberSelectedFeatures, RFE.(RFEtype).(datasetName).accuracyUsingSelectedFeatures.mean,RFE.(RFEtype).(datasetName).accuracyUsingSelectedFeatures.std,'LineWidth',1,'DisplayName',datasetName);
                end
            end
            if contains(datasetName,'Testing')
                figure(hAll(cntRFEtype));
                if strcmp(RFEtype,'singleFeatures')
                    plot(smooth(RFE.(RFEtype).(datasetName).accuracyUsingSelectedFeatures.mean,smooth_span_single_features),'LineWidth',1,'DisplayName',folder_short);
                else
                    errorbar(RFE.(RFEtype).(datasetName).numberSelectedFeatures, RFE.(RFEtype).(datasetName).accuracyUsingSelectedFeatures.mean,RFE.(RFEtype).(datasetName).accuracyUsingSelectedFeatures.std,'LineWidth',1,'DisplayName',folder_short);
                end
                
                if strcmp(RFEtype,'signalBlocks')
                    fprintf('Ranking of signal blocks for %s:\n', folder_short);
                    for cntSignal = 1 : size(RFE.signalBlocks.sortedByRank.Name,1)
                        fprintf('Rank %d: %s\n', cntSignal, RFE.signalBlocks.sortedByRank.Name{cntSignal,:});
                    end
                    fprintf('\n');
                end
            end
        end
        if plot_individual_dataset
            for cnth = 1 : length(h)
                figure(h(cnth));
                legend show
                xlabel('Number of Feature Blocks');
                ylabel('Accuracy in %');
                title(['Supervised RFE ',folder_short, ' - ', RFEtype])
                grid on
                if save_plot
                    save_path = join(folder_split(1:end-1),filesep);
                    save_path = save_path{1};
                    savefig(h(cnth),[save_path,filesep,'RFE_',RFEtype,'.fig']);
                    matlab2tikz('floatFormat','%.2f','figurehandle',h(cnth),'showInfo',false,[save_path,filesep,'RFE_',RFEtype,'.tikz']);
                    saveas(h(cnth),[save_path,filesep,'RFE_',RFEtype,'.emf']);
                end
            end
        end
    end
end

if save_plot
    % Save command window and set diary-mode off
    diary off
end

for cnth = 1 : length(hAll)
    figure(hAll(cnth))
    RFEtype = fieldsRFE{cnth};
    title(['Supervised RFE - ', RFEtype])
    legend show
    xlabel('Number of Feature Blocks');
    ylabel('Accuracy in %');
    grid on
    if save_plot
        savefig([save_path_comparison,filesep,'RFE_SupervisedComparison_',RFEtype,'_test','.fig']);
        matlab2tikz('floatFormat','%.2f','showInfo',false,[save_path_comparison,filesep,'RFE_SupervisedComparison_',RFEtype,'_test','.tikz']);
        saveas(gcf,[save_path_comparison,filesep,'RFE_SupervisedComparison_',RFEtype,'_test','.emf']);
    end
end
