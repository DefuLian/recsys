parpool('local', 10);
addpath(genpath('~/code/recsys'))
dir = '~/data';
datasets = {'yelpdata', 'amazondata', 'ml10Mdata', 'netflixdata'};

alg = @(mat,varargin) dmf(mat, 'K', 64, 'max_iter',20, 'islogit', true, varargin{:});
paras = {'rho',10.^(-6:0), 'alpha', 10.^(-4:2), 'beta',10.^(-4:2)};

file_name = sprintf('%s/dmf_if_results(new_ndcg).mat',dir);
if exist(file_name, 'file')
    load(file_name);
end
if ~exist('result', 'var')
    result = cell(length(datasets),1);
end
for i=1:length(datasets)
    if ~isempty(result{i})
        continue
    end
    dataset = datasets{i};
    fprintf('%s\n', dataset)
    load(sprintf('%s/%s.mat', dir , dataset))
    if ~exist('data','var')
        data = Traindata + Testdata;
    end
    data = explicit2implicit(data);
    [outputs{1:6}] = running(alg, data, 'search_mode', 'seq', paras{:});
    result{i} = outputs;
    save(file_name, 'result');
    clear('data')
end
exit
