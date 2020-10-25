function varargout = figurePlotBackgroundMap(varargin)
    % Add path
    pathUpper = strrep(mfilename('fullpath'),mfilename(),'');
    plotMapPath = strcat(pathUpper,filesep,getSettings('plotmappath',true));
    addpath(genpath(plotMapPath));
    % Plot Map
    PlotMap(varargin{:});
end

