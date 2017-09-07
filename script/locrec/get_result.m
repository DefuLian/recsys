% This script is used for cross validation evaluation.

dataset = 'Beijing'; alpha = 30; reg_i = 50000; reg_1 = 10; eta_i = 5; K = 150; K_geomf = 50;

dataset = 'Gowalla'; alpha = 1000; reg_i = 1000000/2; reg_1 = 50; eta_i = 10; K = 150; K_geomf = 150;

fold_mode = 'i';

data = readContent(sprintf('/home/dlian/data/checkin/%s/data.txt', dataset));
[M, N] = size(data);
item_grid = readContent(sprintf('/home/dlian/data/checkin/%s/item_grids_17.txt', dataset), 'nrows', N);
sigma = -1/log(1e-3);
item_grid(item_grid>0) = exp(-item_grid(item_grid>0).^2./sigma);
item_sim = readContent(sprintf('/home/dlian/data/checkin/%s/item_sim.txt', dataset), 'ncols', N, 'nrows', N);


metric_wals = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 20), +(data>0), 'folds', 5, 'fold_mode', fold_mode);

save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_wals');

metric_iccf_8reg = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 20, 'Y', item_grid,  'reg_i', reg_i*8), +(data>0), 'folds', 5, 'fold_mode', fold_mode);
metric_iccf_4reg = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 20, 'Y', item_grid,  'reg_i', reg_i*4), +(data>0), 'folds', 5, 'fold_mode', fold_mode);
metric_iccf_2reg = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 20, 'Y', item_grid,  'reg_i', reg_i*2), +(data>0), 'folds', 5, 'fold_mode', fold_mode);
metric_iccf = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 20, 'Y', item_grid,  'reg_i', reg_i), +(data>0), 'folds', 5, 'fold_mode', fold_mode);
metric_iccf_half_reg = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 20, 'Y', item_grid,  'reg_i', reg_i/2), +(data>0), 'folds', 5, 'fold_mode', fold_mode);
metric_iccf_quarter_reg = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 20, 'Y', item_grid,  'reg_i', reg_i/4), +(data>0), 'folds', 5, 'fold_mode', fold_mode);

save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_iccf', '-append');
save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_iccf_half_reg', '-append');
save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_iccf_quarter_reg', '-append');

metric_geomf = item_recommend(@(mat) geomf(mat, 'alpha', alpha, 'K', K_geomf, 'max_iter', 10, 'Y', item_grid, 'reg_1', reg_1), +(data>0), 'folds', 5, 'fold_mode', fold_mode);
metric_geomf150 = item_recommend(@(mat) geomf(mat, 'alpha', alpha, 'K', 150, 'max_iter', 10, 'Y', item_grid, 'reg_1', reg_1), +(data>0), 'folds', 5, 'fold_mode', fold_mode);

save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_geomf', '-append');
save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_geomf150', '-append');

metric_graph = item_recommend(@(mat) graph_wals(mat, 'alpha', alpha, 'K', K, 'max_iter', 10, 'item_sim', item_sim, 'eta_i', eta_i), +(data>0), 'folds', 5, 'fold_mode', fold_mode);

save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_graph', '-append');

metric_cb = item_recommend(@identity, +(data>0), 'folds', 5, 'fold_mode', fold_mode, 'Q', NormalizeFea(item_sim, 'p', 1));
save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_cb', '-append');



[~, ~, metric_piccf, ~] = hyperp_search(...
    @(varargin) item_recommend(@piccf, +(data>0), 'folds', 5, 'fold_mode', fold_mode, 'alpha', alpha, 'K', 300, 'Y', item_grid, varargin{:}), ...
    @(metric) metric.recall(1,end), 'reg_i', reg_i*[2, 0.5, 0.25, 0.125]);


save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_piccf', '-append');

metric_irenmf = item_recommend(@(mat) irenmf(mat, 'alpha', alpha, 'K', K, 'item_sim', item_sim, 'itemGroup', item_group), +(data>0), 'test_ratio', 0.2, 'split_mode', fold_mode);
save(sprintf('/home/dlian/data/checkin/%s/result_%s.mat', dataset, fold_mode), 'metric_irenmf', '-append');
