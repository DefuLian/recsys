run('~/software/recsys/setup.m')
datasets = {'yelpdata', 'amazondata', 'ml10Mdata', 'netflixdata'};

load('~/result/tkde_dcf/dmf_results(new_ndcg).mat')
paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
clear('result');


for d=1:length(datasets)
    dataset = datasets{d};
    fprintf('%s\n', dataset)
    file_name = sprintf('~/result/tkde_dcf/two_stage_%s.mat', dataset);
    para = paras{d}; para = cell2struct(para(2:2:end), para(1:2:end),1);
    load(sprintf('~/data/%s.mat', dataset))
    [train, test] = split_matrix(data, 'un', 0.8);
    [P,Q] = dmf(train, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'init', true);
    [B,D] = dmf(train, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'alpha', para.alpha, 'beta', para.beta);
    
    [M, N] = size(train);
    can_num = 10;
    max_iter = 5;
    time = zeros(length(can_num)+1, max_iter);
    result = cell(length(can_num)+1, max_iter);
    for iter=1:max_iter
        if M > 20000
            idx = randi(M, 20000, 1);
            train1 = train(idx,:); test1 = test(idx,:);
            P1 = P(idx,:); B1 = B(idx,:);
        else
            train1 = train; test1 = test;
            P1 = P; B1 = B;
        end
        for i=1:can_num
            fprintf('iter=%d,i=%d\n', iter, i+can_num);
            [result{i, iter}, time(i, iter)] = mip_search(struct('real', P1, 'code', B1), struct('real', Q, 'code', D), train1, test1, 200*i);
        end    
        [result{i+1, iter}, time(i+1, iter)] = mip_search(P1, Q, train1, test1, 200);
    end
    save(file_name, 'result', 'time');
end
