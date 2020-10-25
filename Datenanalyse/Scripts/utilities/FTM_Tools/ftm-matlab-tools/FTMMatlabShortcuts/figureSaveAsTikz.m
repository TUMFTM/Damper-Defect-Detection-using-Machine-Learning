%% Save Figure as Tikz-File
function [] = SaveAsTikz()
    % Check for existing Figures
    if isempty(findobj('Type','Figure')), return, end
    % Add path
    pathUpper = strrep(mfilename('fullpath'),mfilename(),'');
    matlab2tikzPath = strcat(pathUpper,filesep,getSettings('matlab2tikzpath',true));
    addpath(genpath(matlab2tikzPath));
    % Change units
    units = get(gcf, 'Units');
    set(gcf, 'Units', 'Pixel')
    % Get path to save
    [FileName,PathName,~] = uiputfile({'*tex', 'Tex-Datei'},'Name wählen', 'Bild.tex');
    if PathName == 0, return, end
    % Save picture
    matlab2tikz([PathName,filesep,strrep(FileName,'*','')]);
    % Change units back
    set(gcf, 'Units', units)
end

