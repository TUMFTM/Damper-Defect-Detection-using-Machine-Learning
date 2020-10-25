function [settingValue] = getSettings(settingName, ispath)
    % Intialization
    settingValue = [];
    % Check input 
    if nargin==0, return, end
    if nargin==1, ispath = false; end
    % Read in Settings
    filepath = strrep(mfilename('fullpath'),mfilename(),'');
    settingsPath = strcat(filepath,'settings.txt');
    fid=fopen(settingsPath,'r');
    % Search input
    while ~feof(fid)
        settings = fgetl(fid);
        strings = strsplit(settings,'=');
        key = strings{1}; value = strings{2};
        if strcmp(lower(key),lower(settingName))
            settingValue = value; 
            break;
        end
    end
    % Replace path sep
    if ispath
        settingValue = strrep(settingValue,'\',filesep);
    end
    % Close file
    fclose(fid);
end

