function hFigure=plotPredictionAnalysisMap(dataStruct, varargin)
%PLOTPREDICTIONANALYSISMAP Summary of this function goes here
%   Detailed explanation goes here
    
    if find(strcmp(varargin,'subplot'))
        ctrl.subplot = varargin{find(strcmp(varargin,'subplot'))+1};
    else
        ctrl.subplot = 0;
    end
    
    if find(strcmp(varargin,'figure'))
        ctrl.figure = varargin{find(strcmp(varargin,'figure'))+1};
    else
        ctrl.figure = 0;
    end

    % Rotwert errechnet sich aus dem inversen Wert der geschätzten Wahrscheinlichkeit,
    % dass ein Sample der tatsächlichen Klasse angehört
    try
        redvalueProSample = (~strcmp(dataStruct.Prop.predictedClass,dataStruct.Prop.labelIsolation).*(1-dataStruct.Prop.Posterior(logical(dataStruct.labelAsMatrix))));
        greenvalueProSample = (strcmp(dataStruct.Prop.predictedClass,dataStruct.Prop.labelIsolation).*dataStruct.Prop.Posterior(logical(dataStruct.labelAsMatrix)));
        color = [redvalueProSample, greenvalueProSample, zeros(size(redvalueProSample))];
    catch
        color = [zeros(size(dataStruct.Prop.LabelID)), zeros(size(dataStruct.Prop.LabelID)), zeros(size(dataStruct.Prop.LabelID))];
    end
    
    if ctrl.figure == 0
        hFigure = figure;
    else
        hFigure = figure(ctrl.figure);
        hold on
    end
    if ctrl.subplot
        ax1 = subplot(1,2,1);
        hold on
        ax2 = subplot(1,2,2);
        hold on
    else
        ax1 = gca;
        hold on
    end
    
%     for cntObs = 1 : length(color)
%         if ctrl.subplot && ~strcmp(dataStruct.Prop.predictedClass(cntObs),dataStruct.Prop.labelIsolation(cntObs))
%             plot(ax2,dataStruct.Prop.GPS_LONG(cntObs), dataStruct.Prop.GPS_LAT(cntObs), 'x', 'color', color(cntObs,:), 'LineWidth', 2, 'MarkerSize', 5);
%         end
%         plot(ax1,dataStruct.Prop.GPS_LONG(cntObs), dataStruct.Prop.GPS_LAT(cntObs), 'x', 'color', color(cntObs,:), 'LineWidth', 2, 'MarkerSize', 5);
%     end
    plot(ax1,dataStruct.Prop.GPS_LONG(1:end,:), dataStruct.Prop.GPS_LAT(1:end,:), 'kx', 'MarkerSize', 5);

    if ctrl.subplot
        linkaxes([ax1 ax2],'xy');
    end
    
%     % Set Colormap
%     vec10 = [1:-0.01:0]';
%     vec01 = [0.01:0.01:1]';
%     color1 = [vec10, zeros(size(vec10)), zeros(size(vec10)); zeros(size(vec01)), vec01, zeros(size(vec01))];
%     colormap(color1);
%     c = colorbar();
%     cminmax = [-1, 1];
%     caxis(gca, cminmax);
%     c.TicksMode = 'manual';
%     c.TickLabelsMode = 'manual';
%     c.Ticks = [-1 0 1];
%     c.TickLabels = {'100% False','unsure','100% Right'};
%     caxis(gca, cminmax);
    
end

