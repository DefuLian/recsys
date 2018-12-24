% the sample dataset is from user-location-count matrix, used for location
% recommendation
parpool('local',2); % when running evaluation for multiple times, we turn on the parallel computing. In this case you need to set up the parallel computing pool.
                    % if parallel computing is not supported in your
                    % machine, you can substitute parfor in heldout_rec and
                    % crossvalid_rec with for
[train, test] = readData('test/dataset/',1); % read train file and test file from data directory
train = +(train>0); % convert count into binary, since it is observed that this could lead to higher recommendation performance compared to using count
test = +(test>0); % also convert count into binary


[summary, detail, elapsed] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 20), train, 'test', test); 
% iccf, without specifying user feature matrix X, and item feature matrix
% Y, is equivalent to WRMF (K=20, alpha=30, max_iter=20)
% this will return summary and detail statistics of metrics.
% summary, as a structure, include mean and standard deviation of multiple runs
% the results of multiple runs are located in detail as a structure
% the summary metrics as structure include recall, precision, map, mpr, ndcg, auc
% for example, you can access 
%       summary.item_recall(1,20) => recall@20 => 0.0754
%       summary.item_ndcg(1,20) => ndcg@20 => 0.0688
%       summary.item_map(1,20) => map@20 => 0.0256
% elapsed report the time for training and testing
%       training time=15.7546  test time=26.0618

% if you have a file which include all data
% you can load data using the "readContent" where you need to specify the
% file name, with an option indicating the index start with zero
% data = readContent('path/to/file', 'zero_start',true)
data = train + test;
% given this data, we can support automatically data split, where you can
% specify mode for splitting as introduced in the readme.
% you can use cross validation or heldout evaluation 
% the case of using cross validation with 5 folds
[summary, detail, elapsed] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 10), data, 'folds', 5, 'fold_mode', 'un'); 
% the case of using heldout evaluation with 2 times of independent runs
[summary, detail, elapsed] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 10), data, 'test_ratio', 0.2, 'split_mode', 'un', 'times', 2); 
% you can also specify the train ration.
% 0.64 of data for training and 0.2 for testing
[summary, detail, elapsed] = item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 20, 'max_iter', 10), data, 'train_ratio', 0.8, 'test_ratio', 0.2, 'split_mode', 'un'); 

% this toolkit provides an important function for parameter tunning
[opt_para, para_all, result, elapsed] = hyperp_search(...
    @(varargin) item_recommend(@iccf, data, 'test_ratio', 0.2, 'K',  20, 'max_iter',10, varargin{:}), ...
    @(metric) metric.item_recall(1,20), 'alpha', [10,20,30]);

% opt_para: resulting optimal parameter 
% para_all: all candidate for tuning parameter
% result: evaluation results of running with each specified parameter
% elapsed: time for each run, including training and testing
