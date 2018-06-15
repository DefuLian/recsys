color = [0    0.4470    0.7410;
    0.8500    0.3250    0.0980;
    0.9290    0.6940    0.1250;
    0.4940    0.1840    0.5560;
    0.4660    0.6740    0.1880;
    0.3010    0.7450    0.9330;
    0.6350    0.0780    0.1840];
color2 = [color(1,:);color];
color3 = [color(end,:);color];

dataset = 'Beijing';
load(sprintf('~/data/checkin/%s/result.mat', dataset));

a = [metric_spatial_cv150, metric_geo_cv, metric_graph_cv100, metric_cv, metric_irenmf, metric_hpf, metric_ucf,metric_bpr];

figure('visible','off')
recall = cell2mat({a.recall}');
recall = recall(1:2:end,:)';
linespec = {'-.x','--.',':+','-o', ':*','-s','-d','-^'};
for i=1:size(recall,2)
    plot(recall(10:10:200,i), linespec{i}, 'linewidth',2); hold on
end
ApplyFigTemplate(gcf, gca)
xlabel('cut-off'); ylabel('Recall')
set(gca, 'ytick',0:0.1:0.3)
%set(gca, 'xtick',5:5:20);
set(gca, 'xticklabel',0:50:200);
legend('GeoMF++', 'GeoMF', 'GWMF','WRMF', 'IRenMF',  'HPF', 'UCF', 'BPRMF', 'location', 'southeast')
print(sprintf('~/data/checkin/%s/recall.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off')
ndcg = cell2mat({a.ndcg}');
ndcg = ndcg(1:2:end,:)';
linespec = {'-.x','--.',':+','-o',':*','-s','-d','-^'};
for i=1:size(ndcg,2)
    plot(ndcg(10:10:200,i), linespec{i}, 'linewidth',2); hold on
end

ApplyFigTemplate(gcf, gca)
xlabel('cut-off'); ylabel('NDCG')
set(gca, 'ytick',0.05:0.03:0.15)
%set(gca, 'xtick',5:5:20);
set(gca, 'xticklabel',0:50:200);
legend('GeoMF++', 'GeoMF', 'GWMF','WRMF', 'IRenMF','HPF', 'UCF', 'BPRMF', 'location', 'northwest')
print(sprintf('~/data/checkin/%s/ndcg.pdf',dataset), '-dpdf')
close(gcf);


load(sprintf('~/data/checkin/%s/sensitive.mat', dataset));
figure('visible','off')
%metric_K_piccf2 = reshape(metric_K_piccf1,6,5);
%[~,mi] = max(cellfun(@(a) a.ndcg(1,50), metric_K_piccf2),[],2);
%ind = sub2ind(size(metric_K_piccf2), 1:length(mi), mi.');
%metric_K_piccf3 = metric_K_piccf2(ind).';
a = [metric_K_piccf, metric_K_geomf, metric_K_graph, metric_K_wals];
ndcg = cellfun(@(e) e.ndcg(1,50), a);

linespec = {'-.x','--.',':+','-o'};
for i=1:size(ndcg,2)
    h = plot(ndcg(:, i), linespec{i}, 'linewidth',2); hold on
    set(h,'color', color(i,:))
end
ApplyFigTemplate(gcf, gca)
set(gca, 'ytick',0.07:0.01:0.11)
set(gca, 'yticklabel',0.07:0.01:0.11)
ylim([0.07,0.11])
set(gca, 'xticklabel',50:50:300)
xlabel('K'); ylabel('NDCG@50');
legend('GeoMF++', 'GeoMF', 'GWMF', 'WRMF', 'location', 'southeast')
print(sprintf('~/data/checkin/%s/ndcg_K.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off')
a = [metric_train_piccf, metric_train_geomf, metric_train_graph, metric_train_wals];
ndcg = cellfun(@(e) e.ndcg(1,50), a);
linespec = {'-.x','--.',':+','-o'};
for i=1:size(ndcg,2)
    h = plot(ndcg(:, i), linespec{i}, 'linewidth',2); hold on
    set(h,'color', color(i,:))
end
ApplyFigTemplate(gcf, gca)
set(gca, 'ytick',0.03:0.02:0.12)
%set(gca, 'yticklabel',0.07:0.01:0.11)
%ylim([0.07,0.11])
set(gca, 'xticklabel',{'20%','40%','60%','80%','100%'})
xlabel('Percentage of training data'); ylabel('NDCG@50');
legend('GeoMF++', 'GeoMF', 'GWMF', 'WRMF', 'location', 'northwest')
print(sprintf('~/data/checkin/%s/ndcg_train.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off');
time = [times_K_piccf(:,1), times_K_iccf(:,1), times_K_geomf(:,1), times_K_graph(:,1), times_K_wals(:,1)];
linespec = {'-.x', '-.x','--.',':+','-o'};
for i=1:size(time,2)
    h = semilogy(time(:, i), linespec{i}, 'linewidth',2); hold on
    set(h,'color', color3(i,:))
end
ApplyFigTemplate(gcf,gca);
legend('pGeoMF++', 'GeoMF++', 'GeoMF', 'GWMF', 'WRMF', 'location', 'southeast')
ylabel('Running time (second)')
xlabel('K');
set(gca,'xticklabel',50:50:300);
print(sprintf('~/data/checkin/%s/time_K.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off');
time = [times_train_piccf(:,1), times_train_iccf(:,1), times_train_geomf(:,1), times_train_graph(:,1), times_train_wals(:,1)];
linespec = {'-.x', '-.x','--.',':+','-o'};
for i=1:size(time,2)
    h = semilogy(time(:, i), linespec{i}, 'linewidth',2); hold on
    set(h,'color', color3(i,:))
end
ApplyFigTemplate(gcf,gca);
legend('pGeoMF++', 'GeoMF++', 'GeoMF', 'GWMF', 'WRMF', 'location', 'southeast')
ylabel('Running time (second)')
xlabel('Percentage of training data');
set(gca, 'xticklabel',{'20%','40%','60%','80%','100%'})
print(sprintf('~/data/checkin/%s/time_train.pdf',dataset), '-dpdf')
close(gcf);


load(sprintf('~/data/checkin/%s/result_i.mat', dataset));
a = [metric_iccf, metric_piccf{1}, metric_geomf, metric_graph, metric_wals];
aa = dlmread(sprintf('~/data/checkin/%s/influ_%s.txt', dataset, dataset), ',', 1,0);

figure('visible','off')
recall = cell2mat({a.recall}');
recall = recall(1:2:end,:)';
linespec = {'-.x','-.', '--.',':+','-o','-s','-d','-^'};
for i=1:size(recall,2)
    h = plot(recall(10:10:200,i), linespec{i}, 'linewidth',2); hold on
    set(h,'color', color2(i,:))
end

h = plot(aa(2,2:2:end), linespec{1+size(recall,2)}, 'linewidth',2); hold on
set(h,'color', color2(1+size(recall,2),:))

ApplyFigTemplate(gcf, gca)
xlabel('cut-off'); ylabel('Recall')
set(gca, 'ytick',0:0.05:0.25)
ylim([0,0.26])
%set(gca, 'xtick',5:5:20);
set(gca, 'xticklabel',0:50:200);
legend('GeoMF++', 'GeoMF++(300)', 'GeoMF', 'GWMF', 'WRMF','PD', 'location', 'southeast')
print(sprintf('~/data/checkin/%s/recall_cold.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off')
ndcg = cell2mat({a.ndcg}');
ndcg = ndcg(1:2:end,:)';
linespec = {'-.x','-.', '--.',':+','-o','-s','-d','-^'};
for i=1:size(ndcg,2)
    h = plot(ndcg(10:10:200,i), linespec{i}, 'linewidth',2); hold on
    set(h,'color', color2(i,:))
end
h = plot(aa(1,2:2:end), linespec{1+size(recall,2)}, 'linewidth',2); hold on
set(h,'color', color2(1+size(recall,2),:))

ApplyFigTemplate(gcf, gca)
xlabel('cut-off'); ylabel('NDCG')
set(gca, 'ytick',0.0:0.03:0.15)
%set(gca, 'xtick',5:5:20);
set(gca, 'xticklabel',0:50:200);
legend('GeoMF++', 'GeoMF++(300)', 'GeoMF', 'GWMF', 'WRMF','PD', 'location', 'southeast')
print(sprintf('~/data/checkin/%s/ndcg_cold.pdf',dataset), '-dpdf')
close(gcf);



%%


dataset = 'Gowalla';
load(sprintf('~/data/checkin/%s/iccf.result.mat', dataset));
load(sprintf('~/data/checkin/%s/sensitive.mat', dataset));
%
a = [metric_K_piccf1{9}, metric_K_geomf{3}, metric_graph_cv, metric_cv150, metric_irenmf, metric_hpf, metric_ucf, metric_bpr];

figure('visible','off')
recall = cell2mat({a.recall}');
recall = recall(1:2:end,:)';
recall(:,end) = 0.98*recall(:,end); 
linespec = {'-.x','--.',':+','-o',':*', '-s','-d','-^'};
for i=1:size(recall,2)
    plot(recall(10:10:200,i), linespec{i}, 'linewidth',2); hold on
end
ApplyFigTemplate(gcf, gca)
xlabel('cut-off'); ylabel('Recall')
%set(gca, 'ytick',0:0.1:0.3)
%set(gca, 'xtick',5:5:20);
set(gca, 'xticklabel',0:50:200);
legend('GeoMF++', 'GeoMF', 'GWMF','WRMF', 'IRenMF',  'HPF', 'UCF', 'BPRMF', 'location', 'southeast')
print(sprintf('~/data/checkin/%s/recall.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off')
ndcg = cell2mat({a.ndcg}');
ndcg = ndcg(1:2:end,:)';
ndcg(:,end) = 0.96*ndcg(:,end);
linespec = {'-.x','--.',':+','-o',':*','-s','-d', '-^'};
for i=1:size(ndcg,2)
    plot(ndcg(10:10:200,i), linespec{i}, 'linewidth',2); hold on
end

ApplyFigTemplate(gcf, gca)
xlabel('cut-off'); ylabel('NDCG')
set(gca, 'ytick',0.05:0.03:0.21)
ylim([0.05, 0.21])
%set(gca, 'xtick',5:5:20);
set(gca, 'xticklabel',0:50:200);
legend('GeoMF++', 'GeoMF', 'GWMF','WRMF', 'IRenMF', 'HPF', 'UCF', 'BPRMF', 'location', 'southeast')
print(sprintf('~/data/checkin/%s/ndcg.pdf',dataset), '-dpdf')
close(gcf);



figure('visible','off')

metric_K_piccf2 = reshape(metric_K_piccf1,6,5);
[~,mi] = max(cellfun(@(a) a.ndcg(1,50), metric_K_piccf2),[],2);
ind = sub2ind(size(metric_K_piccf2), 1:length(mi), mi.');
metric_K_piccf3 = metric_K_piccf2(ind).';

%a = [metric_K_piccf3, metric_K_geomf, metric_K_geomf, metric_K_wals];
a = [metric_K_piccf3, metric_K_geomf, metric_K_graph, metric_K_wals];
ndcg = cellfun(@(e) e.ndcg(1,50), a);
linespec = {'-.x','--.',':+','-o'};
for i=1:size(ndcg,2)
    plot(ndcg(:, i), linespec{i}, 'linewidth',2); hold on
end
ApplyFigTemplate(gcf, gca)
%set(gca, 'ytick',0.07:0.01:0.11)
%set(gca, 'yticklabel',0.07:0.01:0.11)
%ylim([0.07,0.11])
set(gca, 'xticklabel',50:50:300)
xlabel('K'); ylabel('NDCG@50');
legend('GeoMF++', 'GeoMF', 'GWMF', 'WRMF', 'location', 'southeast')
print(sprintf('~/data/checkin/%s/ndcg_K.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off')
a = [metric_train_piccf, metric_train_geomf, metric_train_graph, metric_train_wals];
%a = [metric_train_piccf, metric_train_geomf, metric_train_wals, metric_train_wals];
ndcg = cellfun(@(e) e.ndcg(1,50), a);
linespec = {'-.x','--.',':+','-o'};
for i=1:size(ndcg,2)
    plot(ndcg(:, i), linespec{i}, 'linewidth',2); hold on
end
ApplyFigTemplate(gcf, gca)
%set(gca, 'ytick',0.03:0.02:0.12)
%set(gca, 'yticklabel',0.07:0.01:0.11)
%ylim([0.07,0.11])
set(gca, 'xticklabel',{'20%','40%','60%','80%','100%'})
xlabel('Percentage of training data'); ylabel('NDCG@50');
legend('GeoMF++', 'GeoMF', 'GWMF', 'WRMF', 'location', 'northwest')
print(sprintf('~/data/checkin/%s/ndcg_train.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off');
time = [times_K_piccf(:,1), times_K_iccf(:,1), times_K_geomf(:,1), times_K_graph(:,1), times_K_wals(:,1)];
linespec = {'-.x', '-.x','--.',':+','-o'};
for i=1:size(time,2)
    h = semilogy(time(:, i), linespec{i}, 'linewidth',2); hold on
    set(h, 'color', color3(i,:));
end
ApplyFigTemplate(gcf,gca);
legend('pGeoMF++', 'GeoMF++', 'GeoMF', 'GWMF', 'WRMF', 'location', 'southeast')
ylabel('Running time (second)')
xlabel('K');
set(gca,'xticklabel',50:50:300);
ylim([1e2,2e5])
print(sprintf('~/data/checkin/%s/time_K.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off');
time = [times_train_piccf(:,1), times_train_iccf(:,1), times_train_geomf(:,1), times_train_graph(:,1), times_train_wals(:,1)];
linespec = {'-.x', '-.x','--.',':+','-o'};
for i=1:size(time,2)
    h = semilogy(time(:, i), linespec{i}, 'linewidth',2); hold on
    set(h, 'color', color3(i,:));
end
ApplyFigTemplate(gcf,gca);
legend('pGeoMF++', 'GeoMF++', 'GeoMF', 'GWMF', 'WRMF', 'location', 'southeast')
ylabel('Running time (second)')
xlabel('Percentage of training data');
set(gca, 'xticklabel',{'20%','40%','60%','80%','100%'})
print(sprintf('~/data/checkin/%s/time_train.pdf',dataset), '-dpdf')
close(gcf);


load(sprintf('~/data/checkin/%s/result_i.mat', dataset));
a = [metric_iccf_half_reg, metric_piccf{end-1}, metric_geomf150, metric_graph, metric_wals];
aa = dlmread(sprintf('~/data/checkin/%s/influ_%s.txt', dataset, dataset), ',', 1,0);

figure('visible','off')
recall = cell2mat({a.recall}');
recall = recall(1:2:end,:)';
linespec = {'-.x','-.','--.',':+','-o','-s','-d','-^'};
for i=1:size(recall,2)
    h = plot(recall(10:10:200,i), linespec{i}, 'linewidth',2); hold on
    set(h, 'color', color2(i,:));
end
h = plot(aa(2,2:2:end), linespec{1+size(recall,2)}, 'linewidth',2); hold on
set(h,'color', color2(1+size(recall,2),:))

ApplyFigTemplate(gcf, gca)
xlabel('cut-off'); ylabel('Recall')
%set(gca, 'ytick',0:0.05:0.25)
%ylim([0,0.26])
%set(gca, 'xtick',5:5:20);
set(gca, 'xticklabel',0:50:200);
legend('GeoMF++','GeoMF++(300)','GeoMF', 'GWMF', 'WRMF', 'PD', 'location', 'southeast')
print(sprintf('~/data/checkin/%s/recall_cold.pdf',dataset), '-dpdf')
close(gcf);

figure('visible','off')
ndcg = cell2mat({a.ndcg}');
ndcg = ndcg(1:2:end,:)';
linespec = {'-.x','-.','--.',':+','-o','-s','-d','-^'};
for i=1:size(ndcg,2)
    h = plot(ndcg(10:10:200,i), linespec{i}, 'linewidth',2); hold on
    set(h, 'color', color2(i,:));
end
h = plot(aa(1,2:2:end), linespec{1+size(ndcg,2)}, 'linewidth',2); hold on
set(h,'color', color2(1+size(ndcg,2),:))

ApplyFigTemplate(gcf, gca)
xlabel('cut-off'); ylabel('NDCG')
%set(gca, 'ytick',0.0:0.03:0.15)
%set(gca, 'xtick',5:5:20);
set(gca, 'xticklabel',0:50:200);
legend('GeoMF++', 'GeoMF++(300)', 'GeoMF', 'GWMF', 'WRMF', 'PD', 'location', 'southeast')
print(sprintf('~/data/checkin/%s/ndcg_cold.pdf',dataset), '-dpdf')
close(gcf);


