function histogramAnalysisCatVar(score, labelAsMatrix, varUnderInspection, variableName, varargin)
%HISTOGRAMANALYSISCATVAR Script for analyzing the classification result
%(probabilities) related to a categorical variable (e. g. Roughness Category)
% histogramAnalysisCatVar(featuresTesting, featuresTesting.Prop.here.RoughnessCat, 'RoughnessCat');
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

    % Convert numeric data to categorical if necessary
    if ~iscategorical(varUnderInspection)
        varUnderInspection = categorical(varUnderInspection);
    end
    
    [nVarUnderInsp, categoriesVarUnderInsp] = histcounts(varUnderInspection);
    
    for cntCat = 1 : length(categoriesVarUnderInsp)
        TrueClassPosterior = max(score .* labelAsMatrix, [], 2);
        TrueClassPosteriorOfCategory = TrueClassPosterior(varUnderInspection==categoriesVarUnderInsp(cntCat));

        [countsTrueClass, edgesBinTrueClass] = histcounts(TrueClassPosteriorOfCategory, [0:0.1:1], 'Normalization', ctrl.normalize);
        centersBinTrueClass = edgesBinTrueClass(1:end-1)+diff(edgesBinTrueClass)/2;
        plot(centersBinTrueClass, countsTrueClass,...
            'DisplayName', [variableName, ' = ', categoriesVarUnderInsp{cntCat}, ' (N = ', num2str(nVarUnderInsp(cntCat)),')']);

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