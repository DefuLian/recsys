%%
[train, test] = readData('/home/dlian/data/checkin/Beijing/',1);
[U,V] = piccf(train>0, 'max_iter',10);
[U1,V1] = iccf(train>0, 'max_iter',10);
%%
dataset = 'Beijing';
data = readContent(sprintf('/home/dlian/data/checkin/%s/data.txt',dataset));
[train, test] = split_matrix(+(data>0), 'un', 0.8);

[I,J,V] = find(train);
dlmwrite(sprintf('/home/dlian/data/checkin/%s/train.tsv',dataset),[I-1,J-1,V], 'delimiter', '\t', 'precision', '%d');
[I,J,V] = find(test);
dlmwrite(sprintf('/home/dlian/data/checkin/%s/test.tsv',dataset),[I-1,J-1,V], 'delimiter', '\t', 'precision', '%d');

%% training beijing data

data = readContent('/home/dlian/data/checkin/Beijing/data.txt');
[M, N] = size(data);
item_grid = readContent('/home/dlian/data/checkin/Beijing/item_grids_17.txt', 'nrows', N);
sigma = -1/log(1e-3);
item_grid(item_grid>0) = exp(-item_grid(item_grid>0).^2./sigma);
[train, test] = split_matrix(+(data>0), 'un', 0.8);
item_sim = readContent('/home/dlian/data/checkin/Beijing/item_sim.txt', 'ncols', N, 'nrows', N);

alg = @(varargin) item_recommend(@(mat) iccf(mat, 'Y', item_grid, 'K', 50, 'max_iter', 20, varargin{:}), train, 'test', test, 'topk', 100);
metric_func = @(metric) metric.recall(1,end);
[para, metric2] = hyperp_search(alg, metric_func, 'reg_i', [1000, 5000, 10000, 50000, 100000]);



[reg_graph, ~, metric_reg_graph, times_reg_graph] = hyperp_search(...
    @(varargin) item_recommend(@graph_wals, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'item_sim', item_sim,  varargin{:}), ...
    @(metric) metric.recall(1,end), 'eta_i', [0.1, 0.5, 1, 5, 10]);

%sensitive_analysis('/home/dlian/data/checkin/Beijing', 30, 50, 50000, 10)
alpha = 30; K = 50; reg_i = 50000; reg_1 = 10;

[~, ~, metric_train_wals, times_train_wals] = hyperp_search(...
    @(varargin) item_recommend(@iccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_wals, ~, metric_K_wals, times_K_wals] = hyperp_search(...
    @(varargin) item_recommend(@iccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);


[~, ~, metric_train_iccf, times_train_iccf] = hyperp_search(...
    @(varargin) item_recommend(@iccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'Y', item_grid, 'reg_i', reg_i, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_iccf, ~, metric_K_iccf, times_K_iccf] = hyperp_search(...
    @(varargin) item_recommend(@iccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'Y', item_grid, 'reg_i', reg_i, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);

[~, ~, metric_train_piccf, times_train_piccf] = hyperp_search(...
    @(varargin) item_recommend(@piccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'Y', item_grid, 'reg_i', reg_i, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_piccf, ~, metric_K_piccf, times_K_piccf] = hyperp_search(...
    @(varargin) item_recommend(@piccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'Y', item_grid, 'reg_i', reg_i, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);

[K_piccf, ~, metric_K_piccf1, times_K_piccf] = hyperp_search1(...
    @(varargin) item_recommend(@piccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'Y', item_grid, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300, 'reg_i', reg_i*[2, 1, 0.5, 0.25, 0.125]);


[~, ~, metric_train_geomf, times_train_geomf] = hyperp_search(...
    @(varargin) item_recommend(@geomf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'Y', item_grid, 'reg_1', reg_1, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_geomf, ~, metric_K_geomf, times_K_geomf] = hyperp_search(...
    @(varargin) item_recommend(@geomf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'Y', item_grid, 'reg_1', reg_1, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);

save('/home/dlian/data/checkin/Beijing/sensitive.mat', 'metric_K_geomf', 'times_K_geomf','K_geomf','metric_train_geomf','times_train_geomf', '-append');


[~, ~, metric_train_graph, times_train_graph] = hyperp_search(...
    @(varargin) item_recommend(@graph_wals, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'item_sim', item_sim, 'eta_i', 5, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_graph, ~, metric_K_graph, times_K_graph] = hyperp_search(...
    @(varargin) item_recommend(@graph_wals, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'item_sim', item_sim, 'eta_i', 5, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);

save('/home/dlian/data/checkin/Beijing/sensitive.mat', 'metric_K_graph', 'times_K_graph','K_graph','metric_train_graph','times_train_graph', '-append');

%metric_cv = item_recommend(@(mat) iccf(mat, 'K', 50, 'max_iter', 20), +(data>0), 'folds', 5);
%metric_spatial_cv = item_recommend(@(mat) iccf(mat, 'Y', item_grid, 'K', 50, 'max_iter', 20, 'reg_i', 50000), +(data>0), 'folds', 5);
%save('/home/dlian/data/checkin/Beijing/iccf.result.mat', 'metric_cv', 'metric_spatial_cv');

%metric_cv100 = item_recommend(@(mat) iccf(mat, 'K', 100, 'max_iter', 20), +(data>0), 'folds', 5);
%metric_spatial_cv100 = item_recommend(@(mat) iccf(mat, 'Y', item_grid, 'K', 100, 'max_iter', 20, 'reg_i', 50000), +(data>0), 'folds', 5);
metric_spatial_cv150 = item_recommend(@(mat) iccf(mat, 'Y', item_grid, 'K', 150, 'max_iter', 20, 'reg_i', 50000), +(data>0), 'folds', 5);

save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_spatial_cv150', '-append');

%metric_geo_cv = item_recommend(@(mat) geomf(mat, 'K', 50, 'max_iter', 10, 'reg_1', 10, 'Y', item_grid), +(data>0), 'folds', 5);
%metric_geo_cv100 = item_recommend(@(mat) geomf(mat, 'K', 100, 'max_iter', 10, 'reg_1', 10, 'Y', item_grid), +(data>0), 'folds', 5);

%save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_cv', 'metric_spatial_cv', 'metric_geo_cv', 'metric_cv100', 'metric_spatial_cv100', 'metric_geo_cv100');

metric_graph_cv100 = item_recommend(@(mat) graph_wals(mat, 'K', 100, 'alpha', alpha, 'eta_i', 5, 'item_sim', item_sim), +(data>0), 'folds', 5);
save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_graph_cv100', '-append');
%user_time = readContent('/home/dlian/data/checkin/Beijing/user_time.txt', 'nrows', M);
%user_time = NormalizeFea(user_time);
%item_time = readContent('/home/dlian/data/checkin/Beijing/item_time.txt', 'nrows', N);
%item_time = NormalizeFea(item_time);
%alg = @(varargin) item_recommend(@(mat) iccf(mat, 'reg_i', 50000, 'Y', [item_grid, item_time], 'X', user_time, 'K', 50, 'max_iter', 20, varargin{:}), train, 'test', test, 'topk', 100);
%metric_func = @(metric) metric.recall(1,end);
%[para, metric] = hyperp_search(alg, metric_func, 'reg_u', [1, 10, 100, 1000, 10000]);
%metric0 = item_recommend(@(mat) iccf(mat, 'X', user_time, 'Y', [item_grid, item_time], 'K', 50, 'max_iter', 20, 'reg_i', 50000,'reg_u', 1), train, 'test', test);
%metric1 = item_recommend(@(mat) iccf(mat, 'X', user_time, 'Y', [item_grid, item_time], 'K', 50, 'max_iter', 20, 'reg_i', 50000,'reg_u', 100), train, 'test', test);
%metric2 = item_recommend(@(mat) iccf(mat, 'X', user_time, 'Y', item_grid, 'K', 50, 'max_iter', 20, 'reg_i', 50000,'reg_u', 1), train, 'test', test);
%metric3 = item_recommend(@(mat) iccf(mat, 'X', user_time, 'Y', item_grid, 'K', 50, 'max_iter', 20, 'reg_i', 50000,'reg_u', 100), train, 'test', test);
%metric4 = item_recommend(@(mat) iccf(mat, 'Y', item_grid, 'K', 50, 'max_iter', 20, 'reg_i', 50000), train, 'test', test);
%metric5 = item_recommend(@(mat) iccf(mat, 'Y', [item_grid, item_time], 'K', 50, 'max_iter', 20, 'reg_i', 50000), train, 'test', test);


%metric = item_recommend(@geomf, train, 'test', test, 'topk',100, 'Y', item_grid);
%metric10_1 = item_recommend(@geomf, train, 'test', test, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 10);

[train, test] = readData('/home/dlian/data/checkin/Beijing', 1);
[M, N] = size(train);
hbeta = dlmread('~/data/checkin/n29763-m41222-k150-batch-hier-vb/hbeta.tsv');
hbeta_mat = zeros(N, 150);
hbeta_mat(hbeta(:,2)+1,:) = hbeta(:,3:end);
htheta = dlmread('~/data/checkin/n29763-m41222-k150-batch-hier-vb/htheta.tsv');
htheta_mat = zeros(M,150);
htheta_mat(htheta(:,2)+1,:) = htheta(:,3:end);
metric_hpf = item_recommend(@identity, train, 'test', test, 'P', htheta_mat, 'Q', hbeta_mat);
save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_hpf', '-append');

metric_ucf = item_recommend(@User_iKNN, train, 'test', test);
save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_ucf', '-append');


PP = dlmread('/home/dlian/data/checkin/Beijing/bprmf.model', '\t', [3,0,3+M*150-1,2]);
P = reshape(PP(:,3),150, M).';
P = [P,ones(M,1)];
b = dlmread('/home/dlian/data/checkin/Beijing/bprmf.model', '\t', [5+M*150,0,4+M*150+N,0]);
QQ = dlmread('/home/dlian/data/checkin/Beijing/bprmf.model', '\t', [6+M*150+N,0,5+M*150+N+N*150,2]);
Q = reshape(QQ(:,3),150, N).';
Q = [Q,b];

metric_bpr = item_recommend(@identity, train, 'test', test, 'P', P, 'Q', Q);
save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_bpr', '-append');


fileID = fopen('/home/dlian/data/checkin/Beijing/items.txt');
items = textscan(fileID, '%f\t%f\t%f\t%s');
fclose(fileID);
[IDX, C]=kmeans([items{2}, items{3}], 50);
clusterInx=+[items{1}, IDX];
group_num = max(clusterInx(:,2));
item_group = sparse(clusterInx(:,1)+1, clusterInx(:,2), true, N, group_num);

metric_irenmf = item_recommend(@(mat) irenmf(mat, 'alpha', alpha, 'K', 150, 'item_sim', item_sim, 'itemGroup', item_group), train, 'test', test);
tic
metric_irenmf0 = item_recommend(@(mat) irenmf(mat, 'alpha', alpha, 'K', 150, 'item_sim', item_sim, 'itemGroup', item_group, 'reg_i', 0.015*0.1), train, 'test', test);
metric_irenmf1 = item_recommend(@(mat) irenmf(mat, 'alpha', alpha, 'K', 150, 'item_sim', item_sim, 'itemGroup', item_group, 'reg_i', 0.015*10), train, 'test', test);
metric_irenmf2 = item_recommend(@(mat) irenmf(mat, 'alpha', alpha, 'K', 150, 'item_sim', item_sim, 'itemGroup', item_group, 'reg_i', 0.015*100), train, 'test', test);
toc

save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_irenmf', '-append');
save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_irenmf0', '-append');
save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_irenmf1', '-append');
save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_irenmf2', '-append');
%% training shanghai data

data = readContent('/home/dlian/data/checkin/Shanghai/data.txt');
[~, N] = size(data);
item_grid = readContent('/home/dlian/data/checkin/Shanghai/item_grids_17.txt', 'nrows', N);
sigma = -1/log(1e-3);
item_grid(item_grid>0) = exp(-item_grid(item_grid>0).^2./sigma);
%item_grid = NormalizeFea(item_grid);

% trim some users and items
[data_trim, rows, cols] = trim_data(data, 20);
Y_trim = item_grid(cols,:);

[train, test] = split_matrix(+(data_trim>0), 'un', 0.8);


alg = @(varargin) item_recommend(@(mat) iccf(mat, 'K', 30, 'max_iter', 10, varargin{:}), train, 'test', test, 'topk', 100);
metric_func = @(metric) metric.recall(1,end);
[alpha, metric] = hyperp_search(alg, metric_func, 'alpha', [30, 50, 100, 200, 500]);


alg = @(varargin) item_recommend(@(mat) iccf(mat, 'alpha', 50, 'Y', Y_trim, 'K', 30, 'max_iter', 10, varargin{:}), train, 'test', test, 'topk', 100);
metric_func = @(metric) metric.recall(1,end);
%[regi, metric2] = hyperp_search(alg, metric_func, 'reg_i', [100, 1000, 5000, 10000, 50000, 100000]);
[regi, metric3] = hyperp_search(alg, metric_func, 'reg_i', [100000, 500000, 1000000, 5000000]);


metric_cv = item_recommend(@(mat) iccf(mat, 'alpha', 50, 'K', 50, 'max_iter', 20), +(data_trim>0), 'folds', 5);
metric_spatial_cv = item_recommend(@(mat) iccf(mat, 'alpha', 50, 'K', 50, 'max_iter', 20, 'Y', Y_trim, 'reg_i', 50000), +(data_trim>0), 'folds', 5);

metric_cv100 = item_recommend(@(mat) iccf(mat, 'alpha', 50, 'K', 100, 'max_iter', 20), +(data_trim>0), 'folds', 5);
metric_spatial_cv100 = item_recommend(@(mat) iccf(mat, 'alpha', 50, 'K', 100, 'max_iter', 20, 'Y', Y_trim, 'reg_i', 50000), +(data_trim>0), 'folds', 5);


save('/home/dlian/data/checkin/Shanghai/iccf.result.mat', 'metric_cv', 'metric_spatial_cv', 'metric_cv100', 'metric_spatial_cv100');

metric_cv150 = item_recommend(@(mat) iccf(mat, 'alpha', 50, 'K', 150, 'max_iter', 20), +(data_trim>0), 'folds', 5);
metric_spatial_cv150 = item_recommend(@(mat) iccf(mat, 'alpha', 50, 'K', 150, 'max_iter', 20, 'Y', Y_trim, 'reg_i', 50000), +(data_trim>0), 'folds', 5);
metric_cv200 = item_recommend(@(mat) iccf(mat, 'alpha', 50, 'K', 200, 'max_iter', 20), +(data_trim>0), 'folds', 5);
metric_spatial_cv200 = item_recommend(@(mat) iccf(mat, 'alpha', 50, 'K', 200, 'max_iter', 20, 'Y', Y_trim, 'reg_i', 50000), +(data_trim>0), 'folds', 5);
save('/home/dlian/data/checkin/Shanghai/result.mat', 'metric_cv', 'metric_spatial_cv', ...
    'metric_cv100', 'metric_spatial_cv100', 'metric_cv150', 'metric_spatial_cv150', 'metric_cv200', 'metric_spatial_cv200');

%% training Gowallal data

data = readContent('/home/dlian/data/checkin/Gowalla/data.txt');
[M, N] = size(data);
item_grid = readContent('/home/dlian/data/checkin/Gowalla/item_grids_17.txt', 'nrows', N);
sigma = -1/log(1e-3);
item_grid(item_grid>0) = exp(-item_grid(item_grid>0).^2./sigma);
item_sim = readContent('/home/dlian/data/checkin/Gowalla/item_sim.txt', 'ncols', N, 'nrows', N);
[train, test] = split_matrix(+(data>0), 'un', 0.8);


alg = @(varargin) item_recommend(@(mat) iccf(mat, 'K', 30, 'max_iter', 10, varargin{:}), train, 'test', test, 'topk', 100);
metric_func = @(metric) metric.recall(1,end);
[alpha, metric] = hyperp_search(alg, metric_func, 'alpha', 1000:500:2000);


alg = @(varargin) item_recommend(@(mat) piccf(mat, 'alpha', 1000, 'Y', item_grid, 'K', 150, 'max_iter', 30, varargin{:}), train, 'test', test, 'topk', 200);
metric_func = @(metric) metric.recall(1,end);
[regi, ~, metric_reg_iccf] = hyperp_search(alg, metric_func, 'reg_i', [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 1500000, 2000000]);

%[reg_graph, ~, metric_reg_graph, times_reg_graph] = hyperp_search(...
[reg_graph1, ~, metric_reg_graph1, times_reg_graph1] = hyperp_search(...
    @(varargin) item_recommend(@graph_wals, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'item_sim', item_sim,  varargin{:}), ...
    @(metric) metric.recall(1,end), 'eta_i', [10, 20, 50]);
    %@(metric) metric.recall(1,end), 'eta_i', [0.1, 0.5, 1, 5, 10]);


metric_cv150 = item_recommend(@(mat) piccf(mat, 'alpha', 1000, 'K', 150, 'max_iter', 20), +(data>0), 'folds', 5);
metric_spatial_cv = item_recommend(@(mat) iccf(mat, 'alpha', 1000, 'Y', item_grid, 'K', 50, 'max_iter', 20, 'reg_i', 1000000), +(data>0), 'folds', 5);
save('/home/dlian/data/checkin/Gowalla/iccf.result.mat', 'metric_cv', 'metric_spatial_cv');
metric_spatial_cv151 = item_recommend(@(mat) piccf(mat, 'alpha', 1000, 'Y', item_grid, 'K', 150, 'max_iter', 20, 'reg_i', 500000/2), +(data>0), 'folds', 5);
save('/home/dlian/data/checkin/Gowalla/iccf.result.mat', 'metric_spatial_cv151', '-append');
metric_geo_cv = item_recommend(@(mat) geomf(mat, 'alpha', 1000, 'K', 50, 'max_iter', 10, 'reg_1', 50, 'Y', item_grid), +(data>0), 'folds', 5);
save('/home/dlian/data/checkin/Gowalla/iccf.result.mat', 'metric_geo_cv', '-append');
metric_graph_cv = item_recommend(@(mat) graph_wals(mat, 'alpha', 1000, 'K', 150, 'max_iter', 10, 'eta_i', 10, 'item_sim', item_sim), +(data>0), 'folds', 5);
save('/home/dlian/data/checkin/Gowalla/iccf.result.mat', 'metric_graph_cv', '-append');

%metric_geo_cv100 = item_recommend(@(mat) geomf(mat, 'K', 100, 'max_iter', 10, 'reg_1', 10, 'Y', item_grid), +(data>0), 'folds', 5);

%save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_cv', 'metric_spatial_cv', 'metric_geo_cv', 'metric_cv100', 'metric_spatial_cv100', 'metric_geo_cv100');


alpha = 1000; K = 50; reg_i = 500000; reg_1 = 50;

%sensitive_analysis('/home/dlian/data/checkin/Gowalla', 1000, 50, 1000000, 10)

[~, ~, metric_train_wals, times_train_wals] = hyperp_search(...
    @(varargin) item_recommend(@iccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_wals, ~, metric_K_wals, times_K_wals] = hyperp_search(...
    @(varargin) item_recommend(@iccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);
save('/home/dlian/data/checkin/Gowalla/sensitive.mat', 'metric_K_wals', 'times_K_wals','K_wals','metric_train_wals','times_train_wals');


[~, ~, metric_train_iccf, times_train_iccf] = hyperp_search(...
    @(varargin) item_recommend(@iccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'Y', item_grid, 'reg_i', reg_i, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_iccf, ~, metric_K_iccf, times_K_iccf] = hyperp_search(...
    @(varargin) item_recommend(@iccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'Y', item_grid, 'reg_i', reg_i, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);
save('/home/dlian/data/checkin/Gowalla/sensitive.mat', 'metric_K_iccf', 'times_K_iccf','K_iccf','metric_train_iccf','times_train_iccf', '-append');

[~, ~, metric_train_piccf, times_train_piccf] = hyperp_search(...
    @(varargin) item_recommend(@piccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'Y', item_grid, 'reg_i', reg_i, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_piccf, ~, metric_K_piccf1, times_K_piccf] = hyperp_search(...
    @(varargin) item_recommend(@piccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'Y', item_grid, 'reg_i', reg_i*0.5, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);

[K_piccf, ~, metric_K_piccf1, times_K_piccf] = hyperp_search1(...
    @(varargin) item_recommend(@piccf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'Y', item_grid, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300, 'reg_i', reg_i*[2, 1, 0.5, 0.25, 0.125]);
%save('/home/dlian/data/checkin/Gowalla/sensitive.mat', 'metric_K_piccf', 'times_K_piccf','K_piccf','metric_train_piccf','times_train_piccf', '-append');
save('/home/dlian/data/checkin/Gowalla/sensitive.mat', 'metric_K_piccf','metric_train_piccf', '-append');

[~, ~, metric_train_geomf, times_train_geomf] = hyperp_search(...
    @(varargin) item_recommend(@geomf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'Y', item_grid, 'reg_1', reg_1, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_geomf, ~, metric_K_geomf, times_K_geomf] = hyperp_search(...
    @(varargin) item_recommend(@geomf, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'Y', item_grid, 'reg_1', reg_1, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);

save('/home/dlian/data/checkin/Gowalla/sensitive.mat', 'metric_K_geomf', 'times_K_geomf','K_geomf','metric_train_geomf','times_train_geomf', '-append');


[~, ~, metric_train_graph, times_train_graph] = hyperp_search(...
    @(varargin) item_recommend(@graph_wals, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'K', K, 'item_sim', item_sim, 'eta_i', 10, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);
[K_graph, ~, metric_K_graph, times_K_graph] = hyperp_search(...
    @(varargin) item_recommend(@graph_wals, +(data>0), 'test_ratio', 0.2, 'alpha', alpha, 'item_sim', item_sim, 'eta_i', 10, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50:50:300);

save('/home/dlian/data/checkin/Gowalla/sensitive.mat', 'metric_K_graph', 'times_K_graph','K_graph','metric_train_graph','times_train_graph', '-append');




[train, test] = readData('/home/dlian/data/checkin/Gowalla', 1);
[M, N] = size(train);
hbeta = dlmread('~/data/checkin/n72953-m131328-k150-batch-hier-vb/hbeta.tsv');
hbeta_mat = zeros(N, 150);
hbeta_mat(hbeta(:,2)+1,:) = hbeta(:,3:end);
htheta = dlmread('~/data/checkin/n72953-m131328-k150-batch-hier-vb/htheta.tsv');
htheta_mat = zeros(M,150);
htheta_mat(htheta(:,2)+1,:) = htheta(:,3:end);
metric_hpf = item_recommend(@identity, train, 'test', test, 'P', htheta_mat, 'Q', hbeta_mat);
save('/home/dlian/data/checkin/Gowalla/iccf.result.mat', 'metric_hpf', '-append');

metric_ucf = item_recommend(@User_iKNN, train, 'test', test);
save('/home/dlian/data/checkin/Gowalla/iccf.result.mat', 'metric_ucf', '-append');

PP = dlmread('/home/dlian/data/checkin/Gowalla/bprmf.model', '\t', [3,0,3+M*150-1,2]);
P = reshape(PP(:,3),150, M).';
P = [P,ones(M,1)];
b = dlmread('/home/dlian/data/checkin/Gowalla/bprmf.model', '\t', [5+M*150,0,4+M*150+N,0]);
QQ = dlmread('/home/dlian/data/checkin/Gowalla/bprmf.model', '\t', [6+M*150+N,0,5+M*150+N+N*150,2]);
Q = reshape(QQ(:,3),150, N).';
Q = [Q,b];

metric_bpr = item_recommend(@identity, train, 'test', test, 'P', P, 'Q', Q);
save('/home/dlian/data/checkin/Gowalla/iccf.result.mat', 'metric_bpr', '-append');


fileID = fopen('/home/dlian/data/checkin/Gowalla/items.txt');
items = textscan(fileID, '%f\t%f\t%f\t%s');
fclose(fileID);
[IDX, C]=kmeans([items{2}, items{3}], 50);
clusterInx=+[items{1}, IDX];
group_num = max(clusterInx(:,2));
item_group = sparse(clusterInx(:,1)+1, clusterInx(:,2), true, N, group_num);
[metric_irenmf,time] = item_recommend(@(mat) irenmf(mat, 'alpha', alpha, 'K', 150, 'item_sim', item_sim, 'itemGroup', item_group), train, 'test', test);
save('/home/dlian/data/checkin/Gowalla/iccf.result.mat', 'metric_irenmf', '-append');
%% 
P = randn(N, 50);
X = item_grid;
U = zeros(size(X,2),50);
U1 = CD(P, U, X, 1, 1e-5);

F = size(X,2);
t = X.' * P;
%mat = X.' * X + reg * speye(F, F);
mat = X.' * X + spdiags(ones(F,1),0, F, F);
U = mat \ t;