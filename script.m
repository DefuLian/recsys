%% training beijing data

data = readContent('/home/dlian/data/checkin/Beijing/data.txt');
[M, N] = size(data);
item_grid = readContent('/home/dlian/data/checkin/Beijing/item_grids_17.txt', 'nrows', N);
sigma = -1/log(1e-3);
item_grid(item_grid>0) = exp(-item_grid(item_grid>0).^2./sigma);
[train, test] = split_matrix(+(data>0), 'un', 0.8);

alg = @(varargin) item_recommend(@(mat) iccf(mat, 'Y', item_grid, 'K', 50, 'max_iter', 20, varargin{:}), train, 'test', test, 'topk', 100);
metric_func = @(metric) metric.recall(1,end);
[para, metric2] = hyperp_search(alg, metric_func, 'reg_i', [1000, 5000, 10000, 50000, 100000]);

metric_cv = item_recommend(@(mat) iccf(mat, 'K', 50, 'max_iter', 20), +(data>0), 'folds', 5);
metric_spatial_cv = item_recommend(@(mat) iccf(mat, 'Y', item_grid, 'K', 50, 'max_iter', 20, 'reg_i', 50000), +(data>0), 'folds', 5);
save('/home/dlian/data/checkin/Beijing/iccf.result.mat', 'metric_cv', 'metric_spatial_cv');

metric_cv100 = item_recommend(@(mat) iccf(mat, 'K', 100, 'max_iter', 20), +(data>0), 'folds', 5);
metric_spatial_cv100 = item_recommend(@(mat) iccf(mat, 'Y', item_grid, 'K', 100, 'max_iter', 20, 'reg_i', 50000), +(data>0), 'folds', 5);

save('/home/dlian/data/checkin/Beijing/iccf.result.mat', 'metric_cv', 'metric_spatial_cv');

metric_geo_cv = item_recommend(@(mat) geomf(mat, 'K', 50, 'max_iter', 10, 'reg_1', 10, 'Y', item_grid), +(data>0), 'folds', 5);
metric_geo_cv100 = item_recommend(@(mat) geomf(mat, 'K', 100, 'max_iter', 10, 'reg_1', 10, 'Y', item_grid), +(data>0), 'folds', 5);

save('/home/dlian/data/checkin/Beijing/result.mat', 'metric_cv', 'metric_spatial_cv', 'metric_geo_cv', 'metric_cv100', 'metric_spatial_cv100', 'metric_geo_cv100');

metric_K_wls = cell(10,2);
metric_K_iccf = cell(10,2);
metric_K_geo = cell(10,2);

for K=50:50:500
    tic; metric_K_wls{K/50,1} = item_recommend(@(mat) iccf(mat, 'K', K, 'max_iter', 20), train, 'test', test); metric_K_wls{K/50,2} = toc;
    tic; metric_K_iccf{K/50,1} = item_recommend(@(mat) iccf(mat, 'K', K, 'max_iter', 20, 'Y', item_grid, 'reg_i', 50000), train, 'test', test); metric_K_iccf{K/50,2} = toc;
    tic; metric_K_geo{K/50,1} = item_recommend(@(mat) geomf(mat, 'K', K, 'max_iter', 10, 'Y', item_grid, 'reg_1', 10), train, 'test', test); metric_K_geo{K/50,2} = toc;
end

metric_train_wls = cell(5,2);
metric_train_iccf = cell(5,2);
metric_train_geo = cell(5,2);

for K=1:5
    sub_train = split_matrix(train, 'en', K/5);
    tic; metric_train_wls{K,1} = item_recommend(@(mat) iccf(mat, 'K', 100, 'max_iter', 20), sub_train, 'test', test); metric_train_wls{K,2} = toc;
    tic; metric_train_iccf{K,1} = item_recommend(@(mat) iccf(mat, 'K', 100, 'max_iter', 20, 'Y', item_grid, 'reg_i', 50000), sub_train, 'test', test); metric_train_iccf{K,2} = toc;
    tic; metric_train_geo{K,1} = item_recommend(@(mat) geomf(mat, 'K', 100, 'max_iter', 10, 'Y', item_grid, 'reg_1', 10), sub_train, 'test', test); metric_train_geo{K,2} = toc;
end

save('/home/dlian/data/checkin/Beijing/result_para.mat', 'metric_K_wls', 'metric_K_iccf', 'metric_K_geo', 'metric_train_wls', 'metric_train_iccf', 'metric_train_geo');


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

%% training Gowallal data

data = readContent('/home/dlian/data/checkin/Gowalla/data.txt');
[M, N] = size(data);
item_grid = readContent('/home/dlian/data/checkin/Gowalla/item_grids_17.txt', 'nrows', N);
sigma = -1/log(1e-3);
item_grid(item_grid>0) = exp(-item_grid(item_grid>0).^2./sigma);

[train, test] = split_matrix(+(data>0), 'un', 0.8);


alg = @(varargin) item_recommend(@(mat) iccf(mat, 'K', 30, 'max_iter', 10, varargin{:}), train, 'test', test, 'topk', 100);
metric_func = @(metric) metric.recall(1,end);
[alpha, metric] = hyperp_search(alg, metric_func, 'alpha', 1000:500:2000);


alg = @(varargin) item_recommend(@(mat) iccf(mat, 'alpha', 1000, 'Y', item_grid, 'K', 30, 'max_iter', 10, varargin{:}), train, 'test', test, 'topk', 100);
metric_func = @(metric) metric.recall(1,end);
[regi, metric2] = hyperp_search(alg, metric_func, 'reg_i', [1000, 5000, 10000, 50000, 100000]);
[regi, metric2] = hyperp_search(alg, metric_func, 'reg_i', [500000, 1000000,1500000,2000000]);


metric_cv = item_recommend(@(mat) iccf(mat, 'alpha', 1000, 'K', 50, 'max_iter', 20), +(data>0), 'folds', 5);
metric_spatial_cv = item_recommend(@(mat) iccf(mat, 'alpha', 1000, 'Y', item_grid, 'K', 50, 'max_iter', 20, 'reg_i', 1000000), +(data>0), 'folds', 5);
save('/home/dlian/data/checkin/Gowalla/iccf.result.mat', 'metric_cv', 'metric_spatial_cv');

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