parpool('local', 2);
addpath(genpath('~/code/recsys'))
dir = '~/data';
datasets = {'yelpdata', 'amazondata', 'ml10Mdata', 'netflixdata'};

alg = @(mat,varargin) dmf(mat, 'K', 64, 'max_iter',20, varargin{:});
paras = {'rho',10.^(-6:0), 'alpha', 10.^(-4:2), 'beta',10.^(-4:2)};

load(sprintf('%s/dmf_results.mat',dir));
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
        Traindata(Testdata>0)=0;
        data = Traindata + Testdata;
    end
    [outputs{1:6}] = running(alg, data, 'search_mode', 'seq', paras{:});
    result{i} = outputs;
    save(sprintf('%s/dmf_results.mat',dir), 'result');
end
exit
