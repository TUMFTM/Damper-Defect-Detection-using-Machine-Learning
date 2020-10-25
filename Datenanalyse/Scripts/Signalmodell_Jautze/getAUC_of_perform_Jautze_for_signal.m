function [output] = getAUC_of_perform_Jautze_for_signal(speedForAnalysis,f_vec,is_defect,ctrl,opts,hROC,hROC_Ref2,cntSpeed,speedName)
%PERFORM_JAUTZE_FOR_SIGNAL Summary of this function goes here
%   Detailed explanation goes here

    ctrl.plot_AUC = 0;
    ctrl.plot_Pxx = 0;
    ctrl.plotHistoFFT = 0;
    [~, ~, AUC, AUC_Ref2, ~] = perform_Jautze_for_signal(speedForAnalysis, f_vec, is_defect, ctrl, opts, hROC, hROC_Ref2, cntSpeed, speedName);
    
    output = 1 - AUC;
    
end

