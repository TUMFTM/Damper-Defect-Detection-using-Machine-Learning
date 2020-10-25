function [] = AddToolBar()
% Check for existing Figures
if isempty(findobj('Type','Figure')), return, end

% Add Path
pathUpper = strrep(mfilename('fullpath'),mfilename(),'');
addpath(pathUpper);

%% Create Icons
iconPath = strcat(pathUpper,filesep,getSettings('IconPath',true));
% Undo und Redo
icon = matlab.ui.internal.toolstrip.Icon.UNDO_16.getIconFile;
% icon = fullfile(iconPath,'greenarrowicon.gif');
[cdataUndo,~] = imread(icon); 
[row,col] =find(sum(cdataUndo,3) == 0);
cdataUndo = double(cdataUndo)/255;
for iInd = 1:length(row), cdataUndo(row(iInd),col(iInd),:) = NaN; end
cdataRedo = cdataUndo(:,end:-1:1,:);
% Extract data
icon = fullfile(iconPath,'plotpicker-stem.gif');
[cdata,map] = imread(icon);
map(find(map(:,1)+map(:,2)+map(:,3)==2)) = NaN;
cdataExtr = ind2rgb(cdata,map);
% SubPlot2Plot
icon = fullfile(iconPath,'HDF_grid.gif');
[cdata,map] = imread(icon);
map(find(map(:,1)+map(:,2)+map(:,3)==2)) = NaN;
cdataSubP = ind2rgb(cdata,map);
% Export Figure
icon = fullfile(iconPath,'HDF_grid.gif');
[cdata,map] = imread(icon);
map(find(map(:,1)+map(:,2)+map(:,3)==2)) = NaN;
cdataExpS = ind2rgb(cdata,map);
% SaveAsTikz
icon = fullfile(iconPath,'pagesicon.gif');
[cdata,map] = imread(icon);
map(find(map(:,1)+map(:,2)+map(:,3)==3)) = NaN;
cdataSaTi = ind2rgb(cdata,map);
% SaveAsPDF
icon = fullfile(iconPath,'pdficon.gif');
[cdata,map] = imread(icon);
map(find(map(:,1)+map(:,2)+map(:,3)==3)) = NaN;
cdataSPDF = ind2rgb(cdata,map);
% ClickableLegend
icon = fullfile(iconPath,'tool_pointer_own.gif');
[cdata,map] = imread(icon);
map(find(map(:,1)+map(:,2)+map(:,3)==2)) = NaN;
cdataClLe = ind2rgb(cdata,map);
% Background Map 
icon = fullfile(iconPath,'TUMLogo.gif');
[cdata,map] = imread(icon);
map(find(map(:,1)+map(:,2)+map(:,3)==2)) = NaN;
cdataTUM = ind2rgb(cdata,map);
% Background Map 
icon = fullfile(iconPath,'webicon.gif');
[cdata,map] = imread(icon);
map(find(map(:,1)+map(:,2)+map(:,3)==2)) = NaN;
cdataMap = ind2rgb(cdata,map);

%% Add Icons
tilesPath  = getSettings('tilespath',true);

uipushtool('Cdata',cdataUndo, 'tooltip','undo',... 
                   'ClickedCallback','uiundo(gcbf,''execUndo'')');
uipushtool('Cdata',cdataRedo, 'tooltip','redo',... 
                   'ClickedCallback','uiundo(gcbf,''execRedo'')');
uipushtool('Cdata',cdataExtr, 'tooltip','Extract Line',... 
                   'ClickedCallback','figureExtractLine');
uipushtool('CData',cdataSPDF, 'tooltip','Save as PDF',...
                   'ClickedCallback','figureSaveAsPDFs');               
uipushtool('CData',cdataExpS, 'tooltip','Export figure with custom styles',...
                   'ClickedCallback','figureExport');                 
uipushtool('CData',cdataSaTi, 'tooltip','Save as Tikz',...
                   'ClickedCallback','figureSaveAsTikz');  
uipushtool('CData',cdataSubP, 'tooltip','Subplot -> Single Plot',...
                   'ClickedCallback','figureSubplot2Plot');   
uipushtool('CData',cdataClLe, 'tooltip','Make Legend Clickable',...
                   'ClickedCallback','figureMakeClickableLegend');                 
uipushtool('CData',cdataTUM, 'tooltip','Make TUM Colors',...
                   'ClickedCallback','figureTUMColors');                 
uipushtool('CData',cdataMap, 'tooltip','Background  Map',...
                   'ClickedCallback',sprintf('figurePlotBackgroundMap(''MapPath'',''%s'',''Scale'',2)',tilesPath));       
end