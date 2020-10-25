%% Add Power-Mode Shortcuts
function [] = AddPowerModeShortcuts()
% Add Path
path = strrep(mfilename('fullpath'),mfilename(),'');
addpath(genpath(path));
pathUpper = getRootPath(path,1);
addpath(pathUpper);

% Init
Shortcuts = {'Balanced', 'Power', 'Save'};    
Category  = 'Power Mode';
Callbacks = {'setPowerMode(''balance'')',...
             'setPowerMode(''power'')',...
             'setPowerMode(''save'')'};
         
% Function Callback
Callbacks  = strcat(['addpath(''',pathUpper,'''), '],Callbacks);
Icons  = {'Standard icon','Standard icon','Standard icon'};
iconPath = getSettings('IconPath',true);
Icons = strcat(pathUpper,filesep,iconPath,filesep,Icons);

% Add Shortcuts
addShortcuts2ToolbarCategory(Shortcuts, Callbacks, Icons, Category)

end


%% Error Message
function []=printErrorMessage(functionName)
    waitfor(errordlg(sprintf('%s konnte nicht im Pfad gefunden werden. Bitte zuerst den Pfad hizufügen.', functionName), 'Fehler'));
end