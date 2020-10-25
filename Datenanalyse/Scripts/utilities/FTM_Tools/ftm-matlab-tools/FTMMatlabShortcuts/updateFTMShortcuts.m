function updataFTMShortcuts()
    % Get path
    pathUpper = strrep(mfilename('fullpath'),mfilename(),'');
    addpath(pathUpper);
    currentPath = pwd;
    
    % Check for
    cd(pathUpper)
    [status,cmdout] = system('git fetch origin && git status');
    if status ~= 0; printError(); return, end
    
    % Check for update
    if regexp(cmdout, 'Your branch is behind')
        [status,cmdout] = system(['cd ', pathUpper, '&& git pull']);
    else
        cd(currentPath)
        return
    end
    if status ~= 0; printError(); return, end
    
    pathInstall = getSettings('installpath',true);

    % Install
    cd(strcat(pathUpper,filesep,pathInstall))
    AddShortcuts(); 
end

function printError()
    opts = struct('WindowStyle','modal','Interpreter','none'); 
    errordlg('Failure while updating.','Error...',opts); 
end
