classdef mapTiles
    
    properties (Access = public)
        originShift = 2*pi*6378137/2.0; 
        initialResolution = 2*pi*6378137/256;
        tileSize = 256;
        zoomLevelMin = 0;
        zoomLevelMax = 19;
    end
    
    methods (Access = public)
        % Coordinate Transformation
        function [mx, my] = pixels2Meters(self, px, py, zoomLevel)
            resolution = self.initialResolution / (2^zoomLevel);
            mx = px*resolution - self.originShift;
            my = py*resolution - self.originShift;
        end
        function [px, py] = meters2Pixels(self, mx, my, zoomLevel)
            resolution = self.initialResolution / (2^zoomLevel);
            px = (mx+self.originShift)/resolution;
            py = (my+self.originShift)/resolution;
        end
        function [lat, lon] = meters2Latlon(self, mx, my)
            lon = (mx/self.originShift)*180.0;
            lat = (my/self.originShift)*180.0;
            lat = 180/pi*(2*atan(exp(lat*pi/180.0))-pi/2.0);
        end
        function [mx, my] = latlon2Meters(self, lat, lon)
                mx = lon * self.originShift / 180.0;
                my = log( tan((90+lat) * pi/360.0)) / (pi/180.0);
                my = my*self.originShift/180.0;
        end
        function [lat, lon] = tileLatLonBounds(self, tx, ty, zoomLevel)
            [minx, miny] = self.pixels2Meters(  tx   *self.tileSize,  ty   *self.tileSize, zoomLevel);
            [maxx, maxy] = self.pixels2Meters( (tx+1)*self.tileSize, (ty+1)*self.tileSize, zoomLevel);
            [lat(1), lon(1)] = self.meters2Latlon(minx, miny);
            [lat(2), lon(2)] = self.meters2Latlon(maxx, maxy);
        end
        function [tx,  ty]  = pixels2Tile(self, px, py)
            tx = max([ceil(px/self.tileSize) - 1,0]);
            ty = max([ceil(py/self.tileSize) - 1,0]); 
        end
        function [txVect, tyVect, lonLim, latLim, latGridVec] = tilesInRange(self, lon, lat, zoomLevel)
            % Find Minimum and Maximum Tiles
            tx = zeros(1,2);
            ty = zeros(1,2);
            [mx, my] = self.latlon2Meters(-lat(2), lon(1));
            [px, py] = self.meters2Pixels(mx, my, zoomLevel);
            [tx(1), ty(1)] = self.pixels2Tile(px, py);
            [mx, my] = self.latlon2Meters(-lat(1), lon(2));
            [px, py] = self.meters2Pixels(mx, my, zoomLevel);
            [tx(2), ty(2)] = self.pixels2Tile(px, py);
            txVect = [tx(1):1:tx(2)];
            tyVect = [ty(1):1:ty(2)];
            numTy = length(tyVect);
            % Get Borders of Tiles
            [latUpperleft , lonUpperleft] = self.tileLatLonBounds(txVect(1), tyVect(1), zoomLevel);
            latLim(2) = -latUpperleft(1); lonLim(1) = lonUpperleft(1);
            [latLowerright , lonLowerright] = self.tileLatLonBounds(txVect(end), tyVect(numTy), zoomLevel);
            latLim(1) = -latLowerright(2); lonLim(2) = lonLowerright(2);
            % Grid Points
            latGridVec = zeros(1,numTy+1);
            latGridVec(1) = latLim(1);
            for i=2:1:numTy
                [latLower, ~] = self.tileLatLonBounds(txVect(1), tyVect(numTy-i+1), zoomLevel);
                latGridVec(i) = -latLower(2);
            end
            latGridVec(numTy+1) = latLim(2);
        end
    end
end

