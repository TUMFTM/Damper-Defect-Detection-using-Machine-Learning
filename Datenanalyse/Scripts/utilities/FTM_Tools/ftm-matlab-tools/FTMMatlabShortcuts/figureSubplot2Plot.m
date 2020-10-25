%% Subplot -> Single Plot
function [] = SubPlot2Plot()
    % Check for existing Figures
    if isempty(findobj('Type','Figure')), return, end
    % Get Axes and Legend of Plot
    obj = gca;
    if verLessThan('Matlab','R2014b')
        L = getappdata(obj,'LegendPeerHandle');
    else
        L = getappdata(obj,'LayoutPeers');
    end
    
    % Create new Figure
    fig = figure;
    if exist('AddToolBar.m')==2, AddToolBar(); end
    
    % Copy data
    copyobj([obj,L],fig);
    
    % Adjust Size
    set(gca,'Parent',fig, 'Position', 'default')
    set(get(gca,'Xlabel'),'FontSize', 'default')
    set(get(gca,'Ylabel'),'FontSize', 'default')
    set(get(gca,'Zlabel'),'FontSize', 'default')
    set(get(gca,'Title'),'FontSize', 'default')
    set(getappdata(gca,'LegendPeerHandle'),'FontSize', 'default')   
end