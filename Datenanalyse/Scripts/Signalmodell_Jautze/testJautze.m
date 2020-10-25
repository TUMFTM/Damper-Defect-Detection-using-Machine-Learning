function [DSKW_corrected_by_speed_testing, AUC_corrected_by_speed_testing, DSKW_testing, DSKW_Ref2_testing, Pxx_testing] = testJautze(dataTesting,opts,ctrl,result_optim,DSKW_corr_factor)
%TESTJAUTZE Summary of this function goes here
%   Detailed explanation goes here

    name_input_variable_dataTesting = inputname(1);

    data_for_analysis = detrendData(dataTesting, opts);
    fprintf('using %d observations of dataTraining as data_for_analysis\n', size(data_for_analysis.SPEED_FL,1));

    % Convert Labels
    is_defect = convert_labels(data_for_analysis);

    % Calculate DSKW
    DSKW_corrected_by_speed_testing = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
    AUC_corrected_by_speed_testing = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
    DSKW_testing = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
    DSKW_Ref2_testing = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
    Pxx_testing = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
    AUC_testing = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
    fpr_testing = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
    tpr_testing = struct('SPEED_FL',[],'SPEED_FR',[],'SPEED_RL',[],'SPEED_RR',[]);
    speedNames = fieldnames(DSKW_testing);
    if ctrl.plot_Pxx
        hPxx_testing = figure('Name','Pxx_testing');
    else
        hPxx_testing = 0;
    end
    if ctrl.plot_AUC
        hROC_testing = figure('Name','ROC_testing');
        hROC_Ref2_testing = figure('Name','ROC_Ref2_testing');
    else
        hROC_testing = 0;
        hROC_Ref2_testing = 0;
    end

    meanAUC = 0;
    reset_plot_AUC = 0;
    for cntSpeed = 1 : length(speedNames)

        speedName = speedNames{cntSpeed};
    %     speedForAnalysis = data_for_analysis.(speedName)';

        f_vec = result_optim.(speedName);
        if ctrl.plot_AUC
            ctrl.plot_AUC = 0;
            reset_plot_AUC = 1;
        end
        [DSKW_testing.(speedName), DSKW_Ref2_testing.(speedName), AUC_testing.(speedName), AUC_Ref2_testing.(speedName), Pxx_testing.(speedName)] = perform_Jautze_for_signal(data_for_analysis.(speedName)', f_vec, is_defect.(speedName), ctrl, opts, hROC_testing, hROC_Ref2_testing, cntSpeed, speedName);
        if reset_plot_AUC
            ctrl.plot_AUC = 1;
            reset_plot_AUC = 0;
        end
        
        DSKW_corrected_by_speed_testing.(speedName) = DSKW_testing.(speedName).*interp1(DSKW_corr_factor.(speedName).Speed, DSKW_corr_factor.(speedName).corr_factor, data_for_analysis.Prop.meanWheelspeed);
        
        % Calculate corrected DSKW values AUC values
        if ctrl.plot_AUC
            figure(hROC_testing);
%             subplot(2,2,cntSpeed);
        end
        [AUC_corrected_by_speed_testing.(speedName), fpr, tpr] = fastAUC(logical(is_defect.(speedName)),DSKW_corrected_by_speed_testing.(speedName),0);
        meanAUC = meanAUC + AUC_corrected_by_speed_testing.(speedName);
        if ctrl.plot_AUC
            plot(fpr, tpr, 'DisplayName', [speedName, ' AUC=', num2str(AUC_corrected_by_speed_testing.(speedName))]);
            hold on
        end
    end
    meanAUC = meanAUC/4;
    if ctrl.plot_AUC
        title([name_input_variable_dataTesting, ' mean AUC = ', num2str(meanAUC)],'Interpreter','None');
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        legend('Location','south east');
    end
    
    if ctrl.plotHistoFFT

        DSKW_to_plot = DSKW_corrected_by_speed_testing;

        plotQuantile = 0.9;     % end of histogram x-axis
        nHistoBars = 100;

        maxHistoBar = quantile(reshape([DSKW_to_plot.SPEED_FL; DSKW_to_plot.SPEED_FR; DSKW_to_plot.SPEED_RL; DSKW_to_plot.SPEED_RR],[],1),plotQuantile);
        edges = 0 : maxHistoBar/nHistoBars : maxHistoBar;

        figure('Name','DSKW Histogram FFT')

        subplot(2,2,1)
        histogram(DSKW_to_plot.SPEED_FL(is_defect.SPEED_FL==1),edges,'FaceColor','r')
        hold on
        histogram(DSKW_to_plot.SPEED_FL(is_defect.SPEED_FL==0),edges,'FaceColor','g')
        legend('defect','intact')
        xlabel('DSKW [-]');
        ylabel('Häufigkeit');
        grid on
        title('DSKW FL');

        subplot(2,2,2)
        histogram(DSKW_to_plot.SPEED_FR(is_defect.SPEED_FR==1),edges,'FaceColor','r')
        hold on
        histogram(DSKW_to_plot.SPEED_FR(is_defect.SPEED_FR==0),edges,'FaceColor','g')
        legend('defect','intact')
        xlabel('DSKW [-]');
        ylabel('Häufigkeit');
        grid on
        title('DSKW FR');

        subplot(2,2,3)
        histogram(DSKW_to_plot.SPEED_RL(is_defect.SPEED_RL==1),edges,'FaceColor','r')
        hold on
        histogram(DSKW_to_plot.SPEED_RL(is_defect.SPEED_RL==0),edges,'FaceColor','g')
        legend('defect','intact')
        xlabel('DSKW [-]');
        ylabel('Häufigkeit');
        grid on
        title('DSKW RL');

        subplot(2,2,4)
        histogram(DSKW_to_plot.SPEED_RR(is_defect.SPEED_RR==1),edges,'FaceColor','r')
        hold on
        histogram(DSKW_to_plot.SPEED_RR(is_defect.SPEED_RR==0),edges,'FaceColor','g')
        legend('defect','intact')
        xlabel('DSKW [-]');
        ylabel('Häufigkeit');
        grid on
        title('DSKW RR');

    end
    
end

