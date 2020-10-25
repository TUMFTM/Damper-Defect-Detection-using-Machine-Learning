function histogramAnalysisContVar_3d(score, labelAsMatrix, varUnderInspection, variableName, varargin)
%HISTOGRAMANALYSISCONTVAR Script for analyzing the classification result
%(probabilities) related to a continuous variable (e. g. mean a_x)
% histogramAnalysisContVar(featuresTesting, featuresTesting.Prop.meanWheelspeed, 'WheelSpeed');
% Options: ... 'normalize', 'count'...

    if find(strcmp(varargin,'nBins'))
        ctrl.nBins = varargin{find(strcmp(varargin,'nBins'))+1};
    else
        ctrl.nBins = 5;
    end
    
    if find(strcmp(varargin,'normalize'))
        ctrl.normalize = varargin{find(strcmp(varargin,'normalize'))+1};
    else
        ctrl.normalize = 'probability';
    end

    if find(strcmp(varargin,'handleAxes'))
        ctrl.hAxes = varargin{find(strcmp(varargin,'handleAxes'))+1};
        axes(ctrl.hAxes);
        hold on
    else
        figure();
        ctrl.hAxes = axes;
        hold on
    end
    
    [nVarUnderInsp, edgesVarUnderInsp, binVarUnderInsp] = histcounts(varUnderInspection,ctrl.nBins);
    uniqueBin = unique(binVarUnderInsp);
    uniqueBin(uniqueBin==0) = [];
    for cntBin = 1 : length(uniqueBin)
        TrueClassPosterior = max(score .* labelAsMatrix, [], 2);
        TrueClassPosteriorOfCategory = TrueClassPosterior(binVarUnderInsp==uniqueBin(cntBin));
        
        [countsTrueClass, edgesBinTrueClass] = histcounts(TrueClassPosteriorOfCategory, [0:0.1:1], 'Normalization', ctrl.normalize);
        centersBinTrueClass = edgesBinTrueClass(1:end-1)+diff(edgesBinTrueClass)/2;
        plot(centersBinTrueClass, countsTrueClass, ...
            'DisplayName', [num2str(edgesVarUnderInsp(cntBin)), ' <= ', variableName, ' < ', num2str(edgesVarUnderInsp(cntBin+1)), ' (N = ', num2str(nVarUnderInsp(cntBin)),')']);
        hold on
    end
    legend('Location','NorthWest');
    try
        MakeClickableLegend;
    catch
        fprintf('MakeClickableLegend not in path\n');
    end
    title([variableName, ' Analysis']);
    xlabel('True Class Posterior Probability');
    ylabel('Percentage of Observations within this Category')

end
