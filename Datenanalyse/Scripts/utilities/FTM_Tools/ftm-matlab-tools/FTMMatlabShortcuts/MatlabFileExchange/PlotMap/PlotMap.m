function varargout = PlotMap(varargin)
% adapted Function from Zohar Bar-Yehuda PlotGoogleMap
% https://de.mathworks.com/matlabcentral/fileexchange/27627-zoharby-plot-google-map
%
% function h = plot_google_map(varargin)
% Plots a google map on the current axes using the Google Static Maps API
%
% USAGE:
% h = plot_google_map(Property, Value,...)
% Plots the map on the given axes. Used also if no output is specified
%
% Or:
% [lonVec latVec imag] = plot_google_map(Property, Value,...)
% Returns the map without plotting it
%
% PROPERTIES:
%    Axis           - Axis handle. If not given, gca is used. (LP)
%    Height (640)   - Height of the image in pixels (max 640)
%    Width  (640)   - Width of the image in pixels (max 640)
%    Scale (2)      - (1/2) Resolution scale factor. Using Scale=2 will
%                     double the resulotion of the downloaded image (up
%                     to 1280x1280) and will result in finer rendering,
%                     but processing time will be longer.
%    MapType        - ('roadmap') Type of map to return. Any of [roadmap, 
%                     satellite, terrain, hybrid]. See the Google Maps API for
%                     more information. 
%    Alpha (1)      - (0-1) Transparency level of the map (0 is fully
%                     transparent). While the map is always moved to the
%                     bottom of the plot (i.e. will not hide previously
%                     drawn items), this can be useful in order to increase
%                     readability if many colors are plotted 
%                     (using SCATTER for example).
%    ShowLabels (1) - (0/1) Controls whether to display city/street textual labels on the map
%    Language       - (string) A 2 letter ISO 639-1 language code for displaying labels in a 
%                     local language instead of English (where available).
%                     For example, for Chinese use:
%                     plot_google_map('language','zh')
%                     For the list of codes, see:
%                     http://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
%    Marker         - The marker argument is a text string with fields
%                     conforming to the Google Maps API. The
%                     following are valid examples:
%                     '43.0738740,-70.713993' (default midsize orange marker)
%                     '43.0738740,-70.713993,blue' (midsize blue marker)
%                     '43.0738740,-70.713993,yellowa' (midsize yellow
%                     marker with label "A")
%                     '43.0738740,-70.713993,tinyredb' (tiny red marker
%                     with label "B")
%    Refresh (1)    - (0/1) defines whether to automatically refresh the
%                     map upon zoom/pan action on the figure.
%    AutoAxis (1)   - (0/1) defines whether to automatically adjust the axis
%                     of the plot to avoid the map being stretched.
%                     This will adjust the span to be correct
%                     according to the shape of the map axes.
%    FigureResizeUpdate (1) - (0/1) defines whether to automatically refresh the
%                     map upon resizing the figure. This will ensure map
%                     isn't stretched after figure resize.
%    APIKey         - (string) set your own API key which you obtained from Google: 
%                     http://developers.google.com/maps/documentation/staticmaps/#api_key
%                     This will enable up to 25,000 map requests per day, 
%                     compared to a few hundred requests without a key. 
%                     To set the key, use:
%                     plot_google_map('APIKey','SomeLongStringObtaindFromGoogle')
%                     You need to do this only once to set the key.
%                     To disable the use of a key, use:
%                     plot_google_map('APIKey','')
%
% OUTPUT:
%    h              - Handle to the plotted map
%
%    lonVect        - Vector of Longidute coordinates (WGS84) of the image 
%    latVect        - Vector of Latidute coordinates (WGS84) of the image 
%    imag           - Image matrix (height,width,3) of the map
%
% EXAMPLE - plot a map showing some capitals in Europe:
%    lat = [48.8708   51.5188   41.9260   40.4312   52.523   37.982];
%    lon = [2.4131    -0.1300    12.4951   -3.6788    13.415   23.715];
%    plot(lon,lat,'.r','MarkerSize',20)
%    plot_google_map
%
% References:
%  http://www.mathworks.com/matlabcentral/fileexchange/24113
%  http://www.maptiler.org/google-maps-coordinates-tile-bounds-projection/
%  http://developers.google.com/maps/documentation/staticmaps/
%
% Acknowledgements:
%  Val Schmidt for his submission of get_google_map.m
%
% Author:
%  Zohar Bar-Yehuda
%
% Version 1.5 - 20/11/2014
%       - Support for MATLAB R2014b
%       - several fixes complex layouts: several maps in one figure, 
%         map inside a panel, specifying axis handle as input (thanks to Luke Plausin)
% Version 1.4 - 25/03/2014
%       - Added the language parameter for showing labels in a local language
%       - Display the URL on error to allow easier debugging of API errors
% Version 1.3 - 06/10/2013
%       - Improved functionality of AutoAxis, which now handles any shape of map axes. 
%         Now also updates the extent of the map if the figure is resized.
%       - Added the showLabels parameter which allows hiding the textual labels on the map.
% Version 1.2 - 16/06/2012
%       - Support use of the "scale=2" parameter by default for finer rendering (set scale=1 if too slow).
%       - Auto-adjust axis extent so the map isn't stretched.
%       - Set and use an API key which enables a much higher usage volume per day.
% Version 1.1 - 25/08/2011

% MapTiles Object
mapTile = mapTiles();

%% Default parametrs
axHandle = gca;
height = 640;
width = 640;
scale = 2;
zoomOffset = 0;
alphaData = 1;
mapPath = [];

%% Handle input arguments
if nargin >= 1
    for idx = 1:2:length(varargin)
        switch lower(varargin{idx})
            case 'mappath'
                mapPath = varargin{idx+1};
            case 'axis'
                axHandle = varargin{idx+1};
            case 'alpha'
                alphaData = varargin{idx+1};
            case 'scale'
                scale = varargin{idx+1};
            case 'zoomoffset'
                zoomOffset = varargin{idx+1};
            otherwise
                error(['Unknown Property: ' varargin{idx}])
        end
    end
end
if isempty(mapPath), return, end

%% Store paramters in axis handle (for auto refresh callbacks)
ud = get(axHandle, 'UserData');
ud.gmap_params = varargin;
set(axHandle, 'UserData', ud);

%% Check Axis Limits of EPSG:900913 
curAxis = axis(axHandle);
% Latitude = y-Axis
if curAxis(3) < -85, curAxis(3) = -85; end
if curAxis(4) > 85,  curAxis(4) = 85;  end
% Longitude = x-Axis
if curAxis(1) < -180, curAxis(1) = -180; end
if curAxis(1) >  180, curAxis(1) = 0;    end
if curAxis(2) >  180, curAxis(2) = 180;  end
if curAxis(2) < -180, curAxis(2) = 0;    end

%% Behaviour for empty Figure
if isempty(axHandle.Children)%isequal(curAxis,[0 1 0 1])
    curAxis = [-180 180 -85 85];
    axHandle.XLim = curAxis(1:2); axHandle.YLim = curAxis(3:4); %pset(axis(curAxis)
end

%% Axis behaviour for Zoom
% adjust current axis limit to avoid strectched maps
[xExtent,yExtent] = mapTile.latlon2Meters(curAxis(3:4),curAxis(1:2));
xExtent = diff(xExtent); % just the size of the span
yExtent = diff(yExtent); 
% get axes aspect ratio
drawnow
unit = get(axHandle, 'Units'); set(axHandle, 'Units', 'pixel'); pixels = get(axHandle,'Position'); set(axHandle, 'Units', unit); 
ratio = pixels(4)/pixels(3);

if xExtent*ratio > yExtent        
    centerX = mean(curAxis(1:2));
    centerY = mean(curAxis(3:4));
    spanX = (curAxis(2)-curAxis(1))/2;
    spanY = (curAxis(4)-curAxis(3))/2;

    % enlarge the Y extent
    spanY = spanY*xExtent*ratio/yExtent; % new span
    if spanY > 85
        spanX = spanX * 85 / spanY;
        spanY = spanY * 85 / spanY;
    end
    curAxis(1) = centerX-spanX;
    curAxis(2) = centerX+spanX;
    curAxis(3) = centerY-spanY;
    curAxis(4) = centerY+spanY;
elseif yExtent > xExtent*ratio

    centerX = mean(curAxis(1:2));
    centerY = mean(curAxis(3:4));
    spanX = (curAxis(2)-curAxis(1))/2;
    spanY = (curAxis(4)-curAxis(3))/2;
    % enlarge the X extent
    spanX = spanX*yExtent/(xExtent*ratio); % new span
    if spanX > 180
        spanY = spanY * 180 / spanX;
        spanX = spanX * 180 / spanX;
    end

    curAxis(1) = centerX-spanX;
    curAxis(2) = centerX+spanX;
    curAxis(3) = centerY-spanY;
    curAxis(4) = centerY+spanY;
end            
% Enforce Latitude constraints of EPSG:900913
if curAxis(3) < -85
    curAxis(3:4) = curAxis(3:4) + (-85 - curAxis(3));
end
if curAxis(4) > 85
    curAxis(3:4) = curAxis(3:4) + (85 - curAxis(4));
end
axis(axHandle, curAxis); % update axis as quickly as possible, before downloading new image
drawnow

%% Calculate zoom level for current axis limits and Check
[xExtent,yExtent] = mapTile.latlon2Meters(curAxis(3:4), curAxis(1:2));
minResX = diff(xExtent) / width;
minResY = diff(yExtent) / height;
minRes = max([minResX minResY]);
tileSize = 256;
initialResolution = 2 * pi * 6378137 / tileSize; % 156543.03392804062 for tileSize 256 pixels
zoomlevel = floor(log2(initialResolution/minRes))+zoomOffset;
if zoomlevel < 0, zoomlevel = 0; end
if zoomlevel > 17,zoomlevel = 17; end

%% Calculate center coordinate in WGS1984
lat = (curAxis(3)+curAxis(4))/2;
lon = (curAxis(1)+curAxis(2))/2;

%% Get Image convert meshgrid to WGS1984
map = [];
while isempty(map) && zoomlevel > 0 
    [map, lonLim, latLim, latGridVec] = allTilesInRange(mapPath, curAxis(1:2), curAxis(3:4), zoomlevel);
    zoomlevel = zoomlevel - 1;
end
if isempty(map), opts = struct('WindowStyle','modal','Interpreter','none'); errordlg('Map could not be found.','Error...',opts); end
map = cast(flipud(map),'double');
% Delete previous map from plot
if isempty(map), return
else
    if nargout <= 1
        curChildren = get(axHandle,'children');
        map_objs = findobj(curChildren,'tag','gmap');
        bd_callback = [];
        for idx = 1:length(map_objs)
            if ~isempty(get(map_objs(idx),'ButtonDownFcn'))
                bd_callback = get(map_objs(idx),'ButtonDownFcn');
            end
        end
        delete(map_objs)
    end

end
% Longitude linear Spaced
lonVec = linspace(lonLim(1), lonLim(2), size(map,2));
% Latitude spaced according to Mercator projection
for i=1:length(latGridVec)-1
    if i>1
        latVecTile = linspace(latGridVec(i), latGridVec(i+1), 257);
        latVecTile = latVecTile(2:end);
    else
        latVecTile = linspace(latGridVec(i), latGridVec(i+1), 256);
    end
    scaleVec = 1./cos(latVecTile/180*pi);
    indexVec = cumtrapz(1/sum(scaleVec)*scaleVec);
    indexVecDist = linspace(0,indexVec(end),256);
    latVec((i-1)*256+1:i*256) = interp1(indexVec, latVecTile, indexVecDist, 'linear');
end
% Adapt resolution to screen size
unit = get(axHandle, 'Units'); set(axHandle, 'Units', 'pixel'); pixels = get(axHandle,'Position'); set(axHandle, 'Units', unit); 
[lonMesh,latMesh] = meshgrid(lonVec,latVec);
uniWidth  = max(scale*pixels(3)*diff(lonLim)/diff(lonVec([1,end])),256*(length(latGridVec)-1));
uniHeight = max(scale*pixels(4)*diff(latLim)/diff(latVec([1,end])),256*(length(latGridVec)-1));
latVect = linspace(latVec(1),latVec(end),uniHeight);
lonVect = linspace(lonLim(1),lonLim(2),uniWidth);
[uniLonMesh,uniLatMesh] = meshgrid(lonVect,latVect);
uniImag = [];
F = makegriddedinterp(lonMesh', latMesh', map(:,:,1).', 'linear','none');
uniImag(:,:,1) = F({uniLonMesh(1,:),uniLatMesh(:,1)})';
F = makegriddedinterp(lonMesh', latMesh', map(:,:,2).', 'linear','none');
uniImag(:,:,2) = F({uniLonMesh(1,:),uniLatMesh(:,1)})';
F = makegriddedinterp(lonMesh', latMesh', map(:,:,3).', 'linear','none');
uniImag(:,:,3) = F({uniLonMesh(1,:),uniLatMesh(:,1)})';



%% Plot map
if nargout <= 1
    % Display image
    hold(axHandle, 'on');
    h = image(lonVect,latVect,uniImag, 'Parent', axHandle);    
    set(axHandle,'YDir','Normal')
    set(h,'tag','gmap', 'AlphaData', alphaData)

    % Add Dummy for pan and zoom
%     h_tmp = image(lonVect([1 end]),latVect([1 end]),zeros(2),'Visible','off', 'Parent', axHandle);
%     set(h_tmp,'tag','gmap')
    
    % Move map to bottom and override zoom
    uistack(h,'bottom')
    axis(axHandle, curAxis)
    if nargout == 1, varargout{1} = h; end

    % Auto Refresh for zoom, pan and figure resize
    figHandle = axHandle;
    while ~strcmpi(get(figHandle, 'Type'), 'figure')
        figHandle = get(figHandle, 'Parent');
    end
    zoomHandle = zoom(axHandle);   
    panHandle = pan(figHandle);  
    set(zoomHandle,'ActionPostCallback',@updateMap);          
    set(panHandle, 'ActionPostCallback', @updateMap);        
    if isempty(get(figHandle, 'ResizeFcn'))
        set(figHandle, 'ResizeFcn', @updateMapFigResize);       
    end    

    % Add ButtonDownFcn
    set(h,'ButtonDownFcn',bd_callback);
else % don't plot, only return map
    varargout{1} = lonVect;
    varargout{2} = latVect;
    varargout{3} = uniImag;
end
end


%% All Tiles in Range
function [map, lonLim, latLim, latVec] = allTilesInRange(mapPath, lon, lat, zoomLevel)
    persistent readPNG
    if isempty(readPNG), readPNG = imformats('png'); readPNG = readPNG.read; end
    
    folderPath = [mapPath,filesep,num2str(zoomLevel)];
    mapTile = mapTiles();
    [xTiles, yTiles, lonLim, latLim, latVec] = mapTile.tilesInRange(lon, lat, zoomLevel);
    % Read all Images and connect them
    for x=1:1:length(xTiles)
        for y=1:1:length(yTiles)
            % Load Image
            mapTemp = [];
            mapColorTemp = [];
            if exist([folderPath,filesep,num2str(xTiles(x)),filesep,num2str(yTiles(y)),'.png']) == 2
                try
                [mapTemp mapColorTemp] = readPNG([folderPath,filesep,num2str(xTiles(x)),filesep,num2str(yTiles(y)),'.png']);
                catch ME
                end
            else, map = []; return, end
            % Convert Image to rgb
            if size(mapTemp,3) == 1
                imag = ind2rgb(mapTemp, mapColorTemp);
            else
                imag = mapTemp;
            end
            if any(any(imag>1))
                imag = cast(imag,'double')./255;
            end
            % Connect Tiles
            map((y-1)*256+1:y*256,(x-1)*256+1:x*256,:) = imag;
        end
    end
end

%% Update Functions
function updateMap(obj,evd)
    % callback function for auto-refresh
    drawnow;
    try
        axHandle = evd.Axes;
    catch ex
        % Event doesn't contain the correct axes. Panic!
        axHandle = gca;
    end
    ud = get(axHandle, 'UserData');
    if isfield(ud, 'gmap_params')
        params = ud.gmap_params;
        PlotMap(params{:});
    end
end
function updateMapFigResize(obj,evd)
    % callback function for auto-refresh
    drawnow;
    axes_objs = findobj(get(gcf,'children'),'type','axes');
    for idx = 1:length(axes_objs)
        if ~isempty(findobj(get(axes_objs(idx),'children'),'tag','gmap'));
            ud = get(axes_objs(idx), 'UserData');
            if isfield(ud, 'gmap_params')
                params = ud.gmap_params;
            else
                params = {};
            end

            % Add axes to inputs if needed
            if ~sum(strcmpi(params, 'Axis'))
                params = [params, {'Axis', axes_objs(idx)}];
            end
            PlotMap(params{:});
        end
    end
end
%% Fast interp2
function F = makegriddedinterp(varargin)
try
    F = griddedInterpolant(varargin{:});
catch gime
    if iscell(varargin{1})
        method = varargin{3};
    else
        method = varargin{4};
    end
    if any(strcmp(gime.identifier,{'MATLAB:griddedInterpolant:NotNdGridErrId', ...
        'MATLAB:griddedInterpolant:NdgridNotMeshgrid2DErrId'}))
        error(message('MATLAB:interp2:InvalidMeshgrid'));
    elseif(strcmp(gime.identifier,'MATLAB:griddedInterpolant:DegenerateGridErrId') && strcmpi(method,'nearest'))
        error(message('MATLAB:interp2:DegenerateGrid'));
    else
        rethrow(gime);
    end
end
end
