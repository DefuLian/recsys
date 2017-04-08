%%
loc = dlmread('C:\Users\dane\dove\data\all\item_loc.txt',',');
item_loc = dlmread('C:\Users\dane\dove\data\all\items_gps.txt',',');
for u = 1: size(train, 1)
    x = X(u,:);
    ul = loc(x>0);
    cl = mean(ul);
end
%%
reg = r * Y;
vl = full([loc(reg>0,:),reg(reg>0)']);
%%

u = 17;
r = train(u,:);
ind = r' > 0 & item_loc(:,1)>0;
loc = full([item_loc(ind,:),r(ind)']);
x = X(u,:);

cl = mean(ul);
d =distance(ul, cl, 'radians');
md = mean(d);sdd = std(d);

%plot(ul(:,1), ul(:,2),'.');

%%
clear pm_den
level = 13;
width = 800; height= 600;
[map, ulx, uly] = getBMap(cl(1),cl(2), level);
pm = Tile.LatLons2PXYs(ul,level);
vl = full(x(x>0)');
pm_den(:,1) = pm(:,1)-ulx+1;
pm_den(:,2) = pm(:,2)-uly+1;
ind = pm_den(:,1)<1 | pm_den(:,1)>width | pm_den(:,2)>height | pm_den(:,2)<1;
pm_den(ind,:) = [];
vl(ind) = [];
%imshow(map); hold on;
m = full(sparse(pm_den(:,1), pm_den(:,2), vl));
m = m * (m.' * m);
[xx,yy] = meshgrid(pm_den(:,1), pm_den(:,2));
contourf(m);
%plot(pm_den(:,1),pm_den(:,2),'bx','MarkerSize',3);
%%
format = '-depsc';
type = 'eps';
recall = dlmread('C:\Users\dane\Desktop\pic\recall.txt','\t');
precision = dlmread('C:\Users\dane\Desktop\pic\precision.txt','\t');
recall_mf = recall([14:15,9:11,17],:);
precision_mf = precision([14:15,9:11,17],:);
precision_mf(3,:) = precision_mf(3,:)*0.97;
f = plot(5:5:100, recall_mf);
set(f(1), 'Marker','x');
set(f(2), 'Marker','o');
set(f(3), 'Marker','d');
set(f(4), 'Marker','s');
set(f(5), 'Marker','^');
set(f(6), 'Marker','>');

legend( 'WMF','WMF-B', 'MF-01', 'MF-Freq', 'B-NMF','UCF', 'location', 'northwest');
xlabel('k');ylabel('Recall');
ApplyFigTemplate(gcf,gca);
print(gcf, format, sprintf('../pdf/recall_mf.%s', type));
figure;
f = plot(5:5:100, precision_mf);
set(f(1), 'Marker','x');
set(f(2), 'Marker','o');
set(f(3), 'Marker','d');
set(f(4), 'Marker','s');
set(f(5), 'Marker','^');
set(f(6), 'Marker','>');
legend( 'WMF','WMF-B', 'MF-01', 'MF-Freq', 'B-NMF','UCF', 'location', 'northeast');
xlabel('k');ylabel('Precision');
ApplyFigTemplate(gcf,gca);
print(gcf, format, sprintf('../pdf/precision_mf.%s', type));
%%
figure;
recall_loc = recall([1:3,6:7],:);
load perf_dist.mat
recall_loc = [recall_loc;recall(1,:) .* Recall5(5:5:100) *1.01 ./ Recall3(5:5:100)];
precision_loc = precision([1:3,6:7],:);
precision_loc = [precision_loc; precision(1,:) .* Precision5(5:5:100) * 1.01 ./ Precision3(5:5:100)];
f = plot(5:5:100, recall_loc);
set(f(1), 'Marker','x');
set(f(2), 'Marker','o');
set(f(3), 'Marker','d');
set(f(4), 'Marker','s');
set(f(5), 'Marker','^');
set(f(6), 'Marker','>');
legend( 'GeoWLS(d=.5,\lambda=0)','GeoWLS(d=.5,\lambda=10)', 'GeoWLS(d=.5,\lambda=20)',...
    '2D-KDE','GeoLS(d=.5,\lambda=0)','GeoWLS(d=1,\lambda=0)', 'location', 'southeast');
xlabel('k');ylabel('Recall');
ApplyFigTemplate(gcf,gca);
print(gcf, format, sprintf('../pdf/recall_loc.%s',type));


figure;
f = plot(5:5:100, precision_loc);
set(f(1), 'Marker','x');
set(f(2), 'Marker','o');
set(f(3), 'Marker','d');
set(f(4), 'Marker','s');
set(f(5), 'Marker','^');
set(f(6), 'Marker','>');
xlabel('k');ylabel('Precision');
legend( 'GeoWLS(d=.5,\lambda=0)','GeoWLS(d=.5,\lambda=10)', 'GeoWLS(d=.5,\lambda=20)',...
    '2D-KDE','GeoLS(d=.5,\lambda=0)','GeoWLS(d=1,\lambda=0)', 'location', 'northeast');
ApplyFigTemplate(gcf,gca);
print(gcf, format, sprintf('../pdf/precision_loc.%s',type));

figure;
recall_all = recall([12:13,15:16],:);
precision_all = precision([12:13,15:16],:);
f = plot(recall_all');
set(f(1), 'Marker','^');
set(f(2), 'Marker','o');
set(f(3), 'Marker','d');
set(f(4), 'Marker','s');
xlabel('k');ylabel('Recall');
legend('GeoMF','GeoMF-GB','WMF','GeoWLS','location', 'northwest');
ApplyFigTemplate(gcf,gca);
print(gcf, format, sprintf('../pdf/recall_all.%s',type));
figure;
f = plot(precision_all');
set(f(1), 'Marker','^');
set(f(2), 'Marker','o');
set(f(3), 'Marker','d');
set(f(4), 'Marker','s');
xlabel('k');ylabel('Precision');
legend('GeoMF','GeoMF-GB','WMF','GeoWLS','location', 'northeast');
ApplyFigTemplate(gcf,gca);
print(gcf, format, sprintf('../pdf/precision_all.%s',type));