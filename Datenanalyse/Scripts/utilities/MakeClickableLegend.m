% partly copied from: http://www.mathworks.com/matlabcentral/fileexchange/21799-clickablelegend-interactive-highlighting-of-data-in-figures
function [] = MakeClickableLegend(ax)
try
    % Check input       
    if nargin < 1, ax = gca; end
    % Get Handles
    legendHandle = legendOfAxes(ax);
    if legendHandle < 1
        waitfor(warndlg('No legend found.', 'Warning...'))
        return
    end
    props=getLegendProps(legendHandle);
    strings = get(legendHandle, 'String');
    % Build newe
    [varargout{1:nargout(@legend)}] = legend(ax, strings); drawnow
    [legHan, legText, legLines] = varargout{1:3};     
    % Reset properties
    for j=1:length(props)/2
        try
            set(legHan, props{j*2-1}, props{j*2});
        end
    end 
    % Make Clickable
    for j=1:length(legLines)
        set(legText(j), 'HitTest', 'on', 'ButtonDownFcn',...
            @(varargin)togglevisibility(legText(j),legLines(j)));
        set(legLines(j), 'HitTest', 'on', 'ButtonDownFcn', ...
            @(varargin)highlightObject(legText(j),legLines(j),legLines(j)));
    end
catch
    waitfor(warndlg('Clickable Legend couldn''t be generated.', 'Warning...'))
end
end

%% Highlight
function togglevisibility(hObject, obj)
    if strcmp(get(hObject, 'FontAngle'), 'italic') % It is on, turn it off
        set(obj, 'HitTest','on','visible','on','handlevisibility','on'); drawnow
        set(hObject, 'Color', [0 0 0],'FontAngle','normal');
    else
        set(obj,'HitTest','off','Visible','off','handlevisibility','off'); drawnow
        set(hObject, 'Color', [0.5 0.5 0.5],'FontAngle','italic');
    end
end
function highlightObject(lTextObj, lMarkerObj, plotObj)
    lw = get(plotObj,'LineWidth');
    if ~iscell(lw), lw = {lw}; end;
    ms = get(plotObj,'MarkerSize');
    if ~iscell(ms), ms = {ms}; end;

    if strcmp(get(lTextObj, 'EdgeColor'),'none') % It is not selected, highlight it
        %set(lTextObj, 'FontWeight', 'bold');
        set(lTextObj, 'EdgeColor', 'k');
        set(plotObj, {'LineWidth', 'MarkerSize'}, [cellfun(@(x)x+2, lw, 'Uniformoutput', false) cellfun(@(x)x+2, ms, 'uniformoutput', false)]);
    else
        %set(lTextObj, 'FontWeight', 'normal');
        set(lTextObj, 'EdgeColor', 'none');
        set(plotObj, {'LineWidth', 'MarkerSize'}, [cellfun(@(x)x-2, lw, 'Uniformoutput', false) cellfun(@(x)x-2, ms, 'uniformoutput', false)]);
    end
end

%% Get Legend
function [legendHandle] = legendOfAxes(ax)
    % Initialization
    legendHandle = [];
    % Check input
    if nargin~=1 || ~ishandle(ax) || ~strcmp(get(ax, 'Type'), 'axes')
        return
    end
    % Get Legend
    if verLessThan('Matlab','R2014b')
        legendHandle = getappdata(ax,'LegendPeerHandle');
    else
        legendHandle = getappdata(ax,'LayoutPeers');
    end
end
%% Get Properties
function [properties]=getLegendProps(leg)
    props = fieldnames(leg);
    for i=1:length(props)
       properties{i*2-1} = props{i}; 
       properties{i*2} = get(leg, props{i}); 
    end
end