function [rootPath] = getRootPath(path,levelUp)
    % Intialization
    rootPath = [];
    % Check input 
    if nargin<1, return, end
    if nargin<2, levelUp = 0; end
    % Shorten path
    indexSep = strfind(path,filesep);
    if isempty(indexSep), return, end
    if indexSep(end) == length(path), levelUp = levelUp+1; end
    rootPath = path(1:indexSep(end-levelUp+1));
end

