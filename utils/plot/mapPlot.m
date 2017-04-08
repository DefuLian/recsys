%%

fileName = 'loc_bj.txt';
%fileName = 'bj_poi_loc.txt';
lm = dlmread(fileName,',');

level = 12;
pm = Tile.LatLons2PXYs(lm,level);

%%
width = 1200;
height = 1000;

if true
    xp = max(pm)-min(pm);
    ll = mean(lm);
    [map,ulx,uly] = getGMap(ll(1),ll(2),level,'width',width,'height',height);
end
%%
clear pm_den;
pm_den(:,1) = pm(:,1)-ulx+1;
pm_den(:,2) = pm(:,2)-uly+1;
pm_den(pm_den(:,1)<1,:) = [];
pm_den(pm_den(:,1)>width,:) = [];
pm_den(pm_den(:,2)>height,:) = [];
pm_den(pm_den(:,2)<1,:) = [];

pos = randperm(length(pm_den));
ind = min(100000,size(pos,2));
pos = pos(1:ind);
pm_den = pm_den(pos,:);
poi = pm_den;
%%
ng = 400;
x = pm_den(:,1);
y = pm_den(:,2);

[dx,xi]= ksdensity(x,'npoints',500);
[dy,yi]= ksdensity(y,'npoints',500);
[xxi,yyi] = meshgrid(xi,yi);
[dxx,dyy] = meshgrid(dx,dy);
dxy = dxx.*dyy; 
contour(xxi,yyi,dxy,100);
figure;
plot(pm_den(:,1),pm_den(:,2),'bx','MarkerSize',3); hold on;
%%

plot(x,y,'bx','MarkerSize',3); hold on;

%%
[b,density,x,y] = kde2d(randn(10000,2));
contourf(x,y,density);
%%
[b,density,x,y] = kde2d(pm_den,200);
density(density < 0)=0;
density = density .^ 0.2;
density(density < 0.01)=0;
min_lat = min(lm(:,1));
max_lat = max(lm(:,1));
min_lon = min(lm(:,2));
max_lon = max(lm(:,2));
x_range=[116.17, 116.581987]; 
y_range=[39.750256, 40.09]; 
%%
[c,h]=contourf(x,y,density);
set(h,'edgecolor','none');
%%
%density = 1./(1+exp(-density));
%imshow(map); hold on;
%
%colormap(flipud(hot));
figure;
%set(gcf,'PaperSize',[30 30]);
%set(gcf,'PaperPosition',[0 0 30 30]);

colormap((hot));
set(gcf,'PaperUnits','points');

pos = get(gcf,'PaperPosition');
set(gcf,'PaperPosition',[0,0,width,height]);
set(gcf, 'PaperSize',[width,height]);
set(gca,'Position',[0,0,1,1]);
%subplot(1,2,1);
%imshow(map); hold on;
plot(pm_den(:,1),pm_den(:,2),'b.','MarkerSize',3); hold on;
%set(gca,'visible','off')
%subplot(1,2,2);
%imshow(map); hold on;
%plot(checkin(:,1),checkin(:,2),'r.','MarkerSize',3)
%contour(x,y,density,100);hold on;
%set(gca,'visible','off')
%set(gca,'XLim',x_range,'YLim',y_range);
%print -dpng map_bj_1.png 
%alpha(0.5);
%%
checkin = pm_den;
%%

contour(xxi,yyi,pdfxy);
%%
[b,density,x,y] = kde2d(pm_den,100);
%index = find(density<0);
%density(index) = 0;
%density = density.^0.4;

%

contour3(x,y,density,50);hold on;
plot(pm_den(:,1),pm_den(:,2),'bx','MarkerSize',3); hold on;
imshow(map); hold on;
%set(gca, 'XLim', [1 800]); 
%set(gca, 'YLim', [1 600]); 

%imshow(map);hold on;


%


%%
[x,y,z]=peaks(45);
 surf(x,y,z);shading interp;
 %alpha(z)
 Amin=-3;Amax=3;
 alim([Amin,Amax])
 alpha('scaled')
 
 %%
data = csvread('loc_density_bj.txt');
len = sqrt(size(data,1));
x = reshape(data(:,1),len,len);
y = reshape(data(:,2),len,len);
v = reshape(data(:,3),len,len);

[c,h]=contourf(y,x,v.^0.5); hold on;
set(h,'edgecolor','none');
print -dmeta density.emf
%imshow('bmap.png');hold on;
%plot(data(:,1),data(:,2),'.');
%print -dpng density.png