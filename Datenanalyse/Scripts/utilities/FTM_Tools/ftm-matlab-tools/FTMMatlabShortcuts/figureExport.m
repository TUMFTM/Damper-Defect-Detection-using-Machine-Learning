function [] = figureExport(ExportStyle, format, path)
%% Save all figures to folder
% This function earches for all open figures and saves them to the
% specified folder. The inputs are strings.
% To define Export settings go to "File-->Export Setup-->Set Style as you
% want --> Save as style named" in the figures window.
% Usefull file formats:
% 'jpeg': JPEG 24 Bit
% 'meta': EMF file (windows only)
% 'svg': Scalable Vecotr Graphics
%
% The function has not output
% Sebastian Wolff, 29.01.2019

% Check input
if nargin==0
    ExportStyle = [];
    format = 'meta';
    path = [];
end
    

% Check mode
handle = gcbo;
if strcmp(class(handle),'matlab.ui.container.toolbar.PushTool')
    handles = handle.Parent.Parent;
else
    handles=findall(0,'type','figure');
end

% Get export styles
if isempty(ExportStyle)
    [exportStyles,exportStylesPath] = getExportStyles();
    [index] = listdlg('PromptString','Choose export style','SelectionMode','single',...
                      'ListString',exportStyles);
end
if isempty(index), return, end

% Load style
load(fullfile(exportStylesPath,filesep,exportStyles{index}))

style.Format = format; %I needed this to make it work but maybe you wont.
%apply style sheet info
% Get save folder
if isempty(path)
    path = uigetdir(userpath,'Select folder to save.');
end
if isnumeric(path) && path==0,return, end

% Save
for iFig = 1:size(handles,1)
    fnam = fullfile(path, sprintf('myfig%i',iFig)); % your file name
    hgexport(handles(iFig),fnam,style);
end

fprintf('Saved %i file(s) to %s\n', iFig, path);

end

function [exportStyle,exportStylesPath] = getExportStyles()
    % Intialization
    exportStyle = [];
    % Get folder and files
    pathUpper = strrep(mfilename('fullpath'),mfilename(),'');
    exportStylesPath = strcat(pathUpper,filesep,getSettings('exportstylespath',true));   
    styles = dir(exportStylesPath);
    % Get all styles
    exportStyle = [];
    for iStyle=1:length(styles)
        if ~isempty(regexp(styles(iStyle).name,'.mat$'))
            exportStyle{end+1}=strrep(styles(iStyle).name,'.mat','');
        end
    end
end