function [bmap,ulx,uly] = getBMap(lat,lon,zoom,varargin)
[width, height] = process_options(varargin, 'width',800, 'height',600);
[px,py] = Tile.LatLon2PXY(lat,lon,zoom);
lx = px - floor(width/2)+1;hx = px + floor(width/2);
ly = py - floor(height/2)+1;hy = py + floor(height/2);
ulx = lx; uly = ly;
[ltx,lty] = Tile.PXY2TXY(lx,ly);
[htx,hty] = Tile.PXY2TXY(hx,hy);
[llx,lly] = Tile.TXY2PXY(ltx,lty);
[hhx,hhy] = Tile.TXY2PXY(htx,hty);
hhx = hhx + 255;
hhy = hhy + 255;
l_mar = int32(lx - llx);
r_mar = int32(hhx - hx);
u_mar = int32(ly - lly);
d_mar = int32(hhy - hy);
bmap = [];
for i = ltx:htx
    tmp = [];
    for j = lty:hty
        m = subGetMap(i,j,zoom);
        tmp = [tmp;m];
    end
    bmap = [bmap,tmp];
end
bmap = bmap(u_mar+1:end-d_mar,l_mar+1:end-r_mar,:);
end

function m = subGetMap(tx,ty,zoom)
key = Tile.TXY2QuadKey(tx,ty,zoom);
url = sprintf('http://r1.tiles.ditu.live.com/tiles/r%s.png?g=66',key);
%url = sprintf('http://h0.ortho.tiles.virtualearth.net/tiles/r%s.png?g=66',key)

[im,m] = imread(url);
m = ind2rgb(im,m);
%m = im;
end