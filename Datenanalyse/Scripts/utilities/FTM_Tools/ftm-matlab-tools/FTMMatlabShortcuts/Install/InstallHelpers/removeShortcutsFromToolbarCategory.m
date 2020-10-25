function removeShortcutsFromToolbarCategory(Shortcuts, Category)
    %% Initialization
    if ~iscell(Shortcuts), Shortcuts = {Shortcuts}; end
    if ~iscell(Category),  Category  = {Category};  end
    
    %% Go through shortcuts
    for iShortcut=1:length(Shortcuts)
        % delete if already exist
        existingShortCuts = com.mathworks.mlwidgets.shortcuts.ShortcutUtils.getShortcutsByCategory(Category{iShortcut});
        if ~isempty(existingShortCuts), existingShortCuts = existingShortCuts.toArray; end
        if checkExistingShortcuts(existingShortCuts,Shortcuts{iShortcut})
            awtinvoke(com.mathworks.mlwidgets.shortcuts.ShortcutUtils,...
                      'removeShortcut', Category{iShortcut}, Shortcuts{iShortcut}); 
        end
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
