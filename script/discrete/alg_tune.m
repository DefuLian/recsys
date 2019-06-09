parpool('local',1);
addpath(genpath('~/software/recsys'))
dir = '~/data';
datasets = {'yelpdata', 'amazondata', 'ml10Mdata', 'netflixdata'};
load ~/result/tkde_dcf/dmf_results(new_ndcg).mat
para = cellfun(@(x) x{3}, result, 'UniformOutput', false);
max_iter = 20;

for iter=1:3
    clear('data');
    clear('result');
    dataset = datasets{iter};
    display(dataset)
    load(sprintf('%s/%s.mat', dir , dataset))
    
    para_dmf = para{iter};
    [train, test] = split_matrix(data, 'un', 0.8);
    alg{1} = @(mat) dmf(mat, 'K', 64, 'max_iter', max_iter, para_dmf{:});
    for a=1:8
        alg{a+1} = @(mat) dmf(mat, 'K', 64, 'max_iter', max_iter, 'alg', 'bcd', 'blocksize', 8*a, para_dmf{:});
    end
    
    %alg{3} = @(mat) dmf(mat, 'K', 64, 'max_iter', max_iter, 'alg', 'bcd', 'blocksize', 16, para_dmf{:});
    %alg{4} = @(mat) dmf(mat, 'K', 64, 'max_iter', max_iter, 'alg', 'bcd', 'blocksize', 32, para_dmf{:});
    %alg{5} = @(mat) dmf(mat, 'K', 64, 'max_iter', max_iter, 'alg', 'bcd', 'blocksize', 64, para_dmf{:});

    filename = sprintf('~/result/tkde_dcf/alg_tune_%s_accurate.mat', dataset);
    if exist(filename, 'file')
        load(filename);
    end

    if ~exist('result', 'var')
        result = cell(length(alg),1);
    end

    for i=1:length(alg)
        if ~isempty(result{i})
            continue
        end
        [outputs{1:3}] = rating_recommend(alg{i}, train, 'test', test);
        result{i} = outputs;
        save(filename, 'result');
    end
end




