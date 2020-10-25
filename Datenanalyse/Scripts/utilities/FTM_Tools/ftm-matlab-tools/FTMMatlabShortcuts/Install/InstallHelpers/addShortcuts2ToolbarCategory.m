function addShortcuts2ToolbarCategory(Shortcuts, Callbacks, Icons, Category)
    %% Initialization
    if ~iscell(Shortcuts), Shortcuts = {Shortcuts}; end
    if ~iscell(Callbacks), Callbacks = {Callbacks}; end
    if ~iscell(Icons),     Icons     = {Icons}; end   
    
    %% Go through shortcuts
    for iShortcut=1:length(Shortcuts)
        % delete if already exist
        removeShortcutsFromToolbarCategory(Shortcuts{iShortcut}, Category);
        
        % check icon
        [~,iconName,ext]=fileparts(Icons{iShortcut});
        if exist(Icons{iShortcut})~=2
            if ~strcmp(iconName, 'Standard icon')
                warning(sprintf('Icon (%s) wasn''t found. Replace by standard.', iconName));
            end
            Icons{iShortcut} = 'Standard icon';
        end
        
        % add to toolbar
        awtinvoke(com.mathworks.mlwidgets.shortcuts.ShortcutUtils,...
                  'addShortcutToBottom',...
                  Shortcuts{iShortcut}, Callbacks{iShortcut}, Icons{iShortcut}, Category,...
                  'Shortcuts',true);
    end
end

function doExist = checkExistingShortcuts(existingShortCuts,Shortcut)
    % Initialization
    doExist = false;
    % Go through existing Shortcuts
    for iShortcut=1:length(existingShortCuts)
        if strcmp(existingShortCuts(iShortcut).toString,Shortcut)
            doExist = true;
            return
        end
    end
end
