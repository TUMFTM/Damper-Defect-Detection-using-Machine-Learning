%% Add Power-Mode Shortcuts
function [] = AddToolBarShortcuts()
% Add Path
path = strrep(mfilename('fullpath'),mfilename(),'');
addpath(genpath(path));
pathUpper = getRootPath(path,1);
addpath(pathUpper);

% Get tiles path
tilesPath  = getSettings('tilespath',true);

% Prepare Shortcuts
Shortcuts = {'AddToolBar', 'Extract Lines', 'Save As PDF', 'Export', 'Save As Tikz', 'SubPlot -> Plot', 'Clickable Legend', 'Make TUM colors', 'Background Map'};    
Category  = 'Figures';
Callbacks = {'AddToolBar',...
             'figureExtractLine',...
             'figureSaveAsPDFs',...
             'figureExport',...
             'figureSaveAsTikz',...
             'figureSubplot2Plot',...
             'figureMakeClickableLegend',...
             'figureTUMColors',...
             sprintf('figurePlotBackgroundMap(''MapPath'',''%s'',''Scale'',2)',tilesPath)};
Callbacks  = strcat(['addpath(''',pathUpper,'''), '],Callbacks);    

% Get icons
Icons = {'Standard icon','plotpicker-stem.gif','pdficon.gif','HDF_grid.gif','pagesicon.gif','HDF_grid.gif','tool_pointer_own.gif','TUMLogo.gif','webicon.gif'};
iconPath = getSettings('IconPath',true);
Icons = strcat(pathUpper,filesep,iconPath,filesep,Icons);

% Add Shortcuts
addShortcuts2ToolbarCategory(Shortcuts, Callbacks, Icons, Category)
end


