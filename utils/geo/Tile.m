classdef Tile
   properties (Constant, GetAccess=private)
      EarthRadius = 6378137;
      MinLatitude = -85.05112878;
      MaxLatitude = 85.05112878;
      MinLongitude = -180;
      MaxLongitude = 180;
   end
   methods(Static, Access=private)
       function c = Clip(n, minValue, maxValue)
            c = min(max(n, minValue), maxValue);
       end
       function ms = MapSize(level)
           ms = bitshift(256, level);
       end
       function gr = GroundResoulution(lat, level)
           lat = Tile.Clip(lat, Tile.MinLatitude, Tile.MaxLatitude);
           gr = cos(lat * pi / 180) * 2 * pi * Tile.EarthRadius / MapSize(level);
       end
   end
   methods(Static)
       function pm=LatLons2PXYs(lm,level)
           pm = lm;
           for i=1:size(pm,1)
               lat = pm(i,1);
               lon = pm(i,2);
               [px,py] = Tile.LatLon2PXY(lat,lon,level);
               pm(i,1) = px;
               pm(i,2) = py;
           end
       end
       function [px,py]=LatLon2PXY(lat,lon,level)
           lat = Tile.Clip(lat, Tile.MinLatitude, Tile.MaxLatitude);
           lon = Tile.Clip(lon, Tile.MinLongitude, Tile.MaxLongitude);

           x = (lon + 180) / 360;
           sinLatitude = sin(lat * pi / 180);
           y = 0.5 - log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * pi);

           mapSize = Tile.MapSize(level);
           px = Tile.Clip(x * mapSize + 0.5, 0, mapSize - 1);
           px = round(px);
           py = Tile.Clip(y * mapSize + 0.5, 0, mapSize - 1);
           py = round(py);
       end
       
       function [lat,lon] = PXY2LatLon(px,py,level)
           mapSize = Tile.MapSize(level);
           x = (Tile.Clip(px, 0, mapSize - 1) / mapSize) - 0.5;
           y = 0.5 - (Tile.Clip(py, 0, mapSize - 1) / mapSize);
           lat = 90 - 360 * atan(exp(-y * 2 * pi)) / pi;
           lon = 360 * x;
       end
       function [tx,ty] = PXY2TXY(px,py)
           tx = floor(px/256);
           ty = floor(py/256);
       end
       function [px,py] = TXY2PXY(tx,ty)
           px = tx*256;
           py = ty*256;
       end
       function key = TXY2QuadKey(tx,ty,level)
           key = zeros(1,level);
           for i=level:-1:1
               digit = 0;
               mask = bitshift(1, i-1);
               if bitand(tx,mask)~=0
                   digit = digit + 1; 
               end
               if bitand(ty,mask)~=0
                   digit = digit + 2;
               end
               key(level-i+1) = digit;
           end
           key = sprintf('%d',key);
       end
       function [tx,ty, level] = QuadKey2TXY(quadkey)
           tx = 0;
           ty = 0;
           level = length(quadkey);
           for i = level:-1:1
               mask = bitshift(1, i - 1);
               c = quadkey(level-i+1);
               switch c
                   case '0'
                       break;
                   case '1'
                       tx = bitor(tx,mask);
                       break;
                   case '2'
                       ty = bitor(ty,mask);
                       break;
                   case '3'
                       tx = bitor(tx,mask);
                       ty = bitor(ty,mask);
                       break;
                   case default
                       break;
               end
           end
       end
   end
end


