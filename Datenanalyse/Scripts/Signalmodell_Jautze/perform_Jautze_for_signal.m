function [tmp_DSKW, tmp_DSKW_Ref2, AUC, AUC_Ref2, tmpPxx] = perform_Jautze_for_signal(speedForAnalysis,f_vec,is_defect,ctrl,opts,hROC,hROC_Ref2,cntSpeed,speedName)
%PERFORM_JAUTZE_FOR_SIGNAL Summary of this function goes here
%   Detailed explanation goes here

    N = pow2(floor(log2(size(speedForAnalysis,1))));
    if N ==1
        N=2;
    end
    f = linspace(0,opts.fs/2, N/2);

    fAus = f_vec(1);
    fRef = f_vec(2);
%     fRef2 = f_vec(3);
    fRef2 = fRef.^2./fAus;
    if fRef2 > f(end-ctrl.neighboring_fft_points)
        fRef2 = f(end-ctrl.neighboring_fft_points);
    end

    if ctrl.eliminate_mean
        speedForAnalysis = speedForAnalysis - mean(speedForAnalysis,1);
    end
    
    [tmpPxx,f] = pwelch(speedForAnalysis, [], [], f, opts.fs);
    
    idx_fAus = find(f>=fAus,1,'first');
    idx_fRef = find(f>=fRef,1,'first');
    idx_fRef2 = find(f>=fRef2,1,'first');
    tmp_DSKW = tmpPxx(idx_fAus,:)./tmpPxx(idx_fRef,:);
    tmp_DSKW_Ref2 = (tmpPxx(idx_fAus,:)./tmpPxx(idx_fRef,:)) ./ (tmpPxx(idx_fRef,:)./tmpPxx(idx_fRef2,:));
    for cntNeighbors = 1 : ctrl.neighboring_fft_points
        tmp_DSKW = tmp_DSKW + tmpPxx(idx_fAus+cntNeighbors,:)./tmpPxx(idx_fRef+cntNeighbors,:);
        tmp_DSKW = tmp_DSKW + tmpPxx(idx_fAus-cntNeighbors,:)./tmpPxx(idx_fRef-cntNeighbors,:);

        tmp_DSKW_Ref2 = tmp_DSKW_Ref2 + (tmpPxx(idx_fAus+cntNeighbors,:)./tmpPxx(idx_fRef+cntNeighbors,:)) ./ (tmpPxx(idx_fRef+cntNeighbors,:)./tmpPxx(idx_fRef2+cntNeighbors,:));
        tmp_DSKW_Ref2 = tmp_DSKW_Ref2 + (tmpPxx(idx_fAus-cntNeighbors,:)./tmpPxx(idx_fRef-cntNeighbors,:)) ./ (tmpPxx(idx_fRef-cntNeighbors,:)./tmpPxx(idx_fRef2-cntNeighbors,:));
    end
    tmp_DSKW = tmp_DSKW/(2*ctrl.neighboring_fft_points+1);
    tmp_DSKW = tmp_DSKW';

    tmp_DSKW_Ref2 = tmp_DSKW_Ref2/(2*ctrl.neighboring_fft_points+1);
    tmp_DSKW_Ref2 = tmp_DSKW_Ref2';

    % Calculate AUC values
    if ctrl.plot_AUC
        figure(hROC);
        subplot(2,2,cntSpeed);
    end
    [AUC, fpr, tpr] = fastAUC(logical(is_defect),tmp_DSKW,ctrl.plot_AUC);
    if ctrl.plot_AUC
        title([speedName, ' AUC = ', num2str(AUC)],'Interpreter','None');
    end
    
    if ctrl.plot_AUC
        figure(hROC_Ref2);
        subplot(2,2,cntSpeed);
    end

    [AUC_Ref2, fpr, tpr] = fastAUC(logical(is_defect),tmp_DSKW_Ref2,ctrl.plot_AUC);

    if ctrl.plot_AUC
        title([speedName, ' AUC = ', num2str(AUC_Ref2)],'Interpreter','None');
    end

end

