%% Add Power-Mode Shortcuts
function [] = AddShortcuts()
%% Add Path
path = strrep(mfilename('fullpath'),mfilename(),'');
addpath(genpath(path));
pathUpper = getRootPath(path,1);
addpath(pathUpper);

%% Init Often Used
Shortcuts = {'CleanUp', 'Broom', 'Open Explorer', 'Update', 'Run', 'New GoTo'};    
Category  = 'Often Used';
Callbacks = {'clc,close all',...
             'clc, close all, clear all',...
             'dos(sprintf(''explorer %s'',pwd));',...
             'updateFTMShortcuts()',...
             ['jDesktop = com.mathworks.mde.desk.MLDesktop.getInstance; ',...
              'jName = jDesktop.getGroupContainer(''Editor'').getComponent(0).getComponent(0).getComponent(0).getName; ',...
              'jName = char(jName) ; eval(strrep(strrep(jName,''Select'',''''),''.mButton'','''')); ',...
              'clear jDesktop jName}; '],...
             ['[~, Folder, ~] = fileparts(pwd); ',...
              'awtinvoke(com.mathworks.mlwidgets.shortcuts.ShortcutUtils,',...
              '''addShortcutToBottom'', Folder, sprintf(''cd(''''%s'''')'',pwd), ''Standard icon'', ''GoTo'',',...
              '''Shortcuts'',true);']};
Icons  = {'HDF_VGroup.gif','paintbrush.gif','upfolder.gif','rating_full.gif','greenarrowicon.gif','tool_arrow.gif'};
iconPath = getSettings('IconPath',true);
Icons = strcat(pathUpper,filesep,iconPath,filesep,Icons);
 

%Add Shortcuts
addShortcuts2ToolbarCategory(Shortcuts, Callbacks, Icons, Category)

%% Init Path
Shortcuts = {'Add', 'Remove', 'Userpath', 'Add permanent', 'Remove permanent'};    
Category  = 'Path';
Callbacks = {'addpath(genpath(pwd))',...
             'rmpath(genpath(pwd))',...
             ['try, userpath(''clear''), userpath(pwd), end, ',...
              'disp(sprintf(''\n\nNeuer Userpath: %s\n\n'', pwd))'],...
              'addpath(pwd), savepath',...
              'try, rmpath(pwd), end, savepath'};
Icons  = {'Standard icon','Standard icon','Standard icon','Standard icon','Standard icon'};
Icons = strcat(pathUpper,filesep,iconPath,filesep,Icons);
 

%Add Shortcuts
addShortcuts2ToolbarCategory(Shortcuts, Callbacks, Icons, Category)

% Add Toolbar Shortcuts
AddToolBarShortcuts()

% PowerMode
AddPowerModeShortcuts()
    
% Remove Shortcuts permanent
Category = {};
Callback = {};

% Add Path to settings
pathAddPermanent = getRootPath(path,1);
rmpath(genpath(path));
rmpath(pathUpper);
addpath(pathAddPermanent)
savepath
end



