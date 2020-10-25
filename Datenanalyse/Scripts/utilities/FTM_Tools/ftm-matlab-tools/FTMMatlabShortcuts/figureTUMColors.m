function figureTUMcolors(ax,values)
    % Check input
    if nargin < 1, if isempty(findobj('type','axes')), return, else, ax = gca; end, end
    if nargin < 2, values = []; end
    % Get Colors
    pathUpper = strrep(mfilename('fullpath'),mfilename(),'');
    tumColorsPath = strcat(pathUpper,filesep,getSettings('tumcolorspath',true));
    addpath(tumColorsPath);
    colors = tumColors;
    % LineStyle
    lineStyle{1} = '-';
    lineStyle{2} = '--';
    lineStyle{3} = ':';
    lineStyle{4} = '-.';
    % Get Number of Children
    numChildren = length(ax.Children);
    for iChildren=1:numChildren
        % Get color
        if isempty(values)
            colorIndex = iChildren;
        else
            colorIndex = values(iChildren);
        end
        % Cut index
        lineStyleIndex = ceil(colorIndex/size(colors.Colormap.Custom,1));
        colorIndex = mod(colorIndex,size(colors.Colormap.Custom,1)); 
        if colorIndex == 0; colorIndex = size(colors.Colormap.Custom,1); end
        color = colors.Colormap.Custom(colorIndex,:);
        lineStyleIndex = mod(lineStyleIndex, length(lineStyle));
        if lineStyleIndex==0, lineStyleIndex = length(lineStyle); end
        % Change Color
        plotIndex = numChildren-iChildren+1;
        if strcmp(ax.Children(plotIndex).Type, 'line')
            ax.Children(plotIndex).Color = color;
            ax.Children(plotIndex).MarkerFaceColor = color;
            ax.Children(plotIndex).LineStyle = lineStyle{lineStyleIndex};
        elseif strcmp(ax.Children(plotIndex).Type, 'bar')
            ax.Children(plotIndex).FaceColor = color;
        end
    end
end
