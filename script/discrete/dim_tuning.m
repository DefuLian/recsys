parpool('local', 6);
addpath(genpath('~/code/recsys'))
dir = '~/data';
datasets = {'yelpdata', 'amazondata', 'ml10Mdata', 'netflixdata'};

load('~/result/dmf_results(new_ndcg).mat');
paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
clear('result');

file_name = sprintf('%s/dim_tuning_results.mat',dir);
if exist(file_name, 'file')
    load(file_name);
end
if ~exist('result', 'var')
    result = cell(length(datasets),2);
end
metric_fun = @(metric) metric.item_ndcg_score(1,end);
for i=1:length(datasets)
    if ~isempty(result{i, 1})
        continue
    end
    
    dataset = datasets{i};
    fprintf('%s\n', dataset)
    load(sprintf('%s/%s.mat', dir , dataset))
    if ~exist('data','var')
        Traindata(Testdata>0)=0;
        data = Traindata + Testdata;
    end
    
    para = paras{i}; para = cell2struct(para(2:2:end), para(1:2:end),1);
    alg = @(mat,varargin) dmf(mat, 'max_iter',20, 'rho', para.rho, 'alpha', para.alpha, 'beta', para.beta, varargin{:});
    [outputs{1:4}] = hyperp_search(...
            @(varargin) rating_recommend(alg, data, 'test_ratio', 0.2, varargin{:}), metric_fun, 'K', [16,32,64,128,256]);
    result{i,1} = outputs;
    
    alg = @(mat,varargin) dmf(mat, 'max_iter',20, 'rho', para.rho, 'alpha', 0, 'beta', 0, varargin{:});
    [outputs{1:4}] = hyperp_search(...
            @(varargin) rating_recommend(alg, data, 'test_ratio', 0.2, varargin{:}), metric_fun, 'K', [16,32,64,128,256]);
    result{i,2} = outputs;
    
    save(file_name, 'result');
    clear('data')
end
exit
