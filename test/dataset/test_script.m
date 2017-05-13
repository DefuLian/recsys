[train, test] = readData('/home/dlian/data/subcheckin/',1);
train = +(train>0); test = +(test>0);
[metric, time] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 20), train, 'test', test); % equivalent to  WRMF
data = train + test;
[metric, time] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 20), data, 'folds', 5, 'fold_mode', 'un'); % using cross validation
[metric, time] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 20), data, 'test_ratio', 0.2, 'split_mode', 'un'); % using holdout evaluation
[metric, time] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 20), data, 'train_ratio', 0.8, 'test_ratio', 0.2, 'split_mode', 'un'); % 0.64 of data for training and 0.2 for testing

[optpar, pars, metric, times] = hyperp_search(...
    @(varargin) item_recommend(@iccf, data, 'test_ratio', 0.2, 'alpha', 30, 'K',  20, varargin{:}), ...
    @(metric) metric.recall(1,end), 'train_ratio', (0.2:0.2:1)*0.8);

[optpar, pars, metric, times] = hyperp_search(...
    @(varargin) item_recommend(@iccf, data, 'test_ratio', 0.2, 'alpha', 30, varargin{:}), ...
    @(metric) metric.recall(1,end), 'K', 50*(1:6));