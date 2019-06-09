
run('~/software/recsys/setup.m')
load('~/result/tkde_dcf/dmf_results(new_ndcg).mat')
paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
para = paras{2}; para = cell2struct(para(2:2:end), para(1:2:end),1);
clear('result');


file_name = '~/result/tkde_dcf/two_stage_dim.mat';
data = readContent('~/data/amazon/ratings_mapped.csv', 'sep', ',');
data = trim_data(data, 10);
max_iter = 5;
ks = [32, 64, 128, 256, 512];
result = cell(length(ks), max_iter+2);
time = zeros(length(ks), max_iter+1);
[train, test] = split_matrix(data, 'un', 0.8);
[M, N] = size(train);
for k = 1:length(ks)
    fprintf('%d\n',ks(k));
    [P,Q] = dmf(train, 'K', ks(k), 'max_iter', 20, 'rho', para.rho, 'init', true);
    [B,D] = dmf(train, 'K', ks(k), 'max_iter', 20, 'rho', para.rho, 'alpha', para.alpha, 'beta', para.beta);
    if M > 20000
        idx = randi(M, 20000, 1);
        train1 = train(idx,:); test1 = test(idx,:);
        P1 = P(idx,:); B1 = B(idx,:);
    else
        train1 = train; test1 = test;
        P1 = P; B1 = B;
    end
    
    for iter=1:max_iter
        [result{k, iter}, time(k, iter)] = mip_search(struct('real', P1, 'code', B1), struct('real', Q, 'code', D), train1, test1, 1000);    
    end
    [result{k, iter+1}, time(k, iter+1)] = mip_search(P1, Q, train1, test1, 200);
    result{k, iter+2} = ks(k);
    
    save(file_name, 'result', 'time');
end



