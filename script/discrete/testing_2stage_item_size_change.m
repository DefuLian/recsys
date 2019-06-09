run('~/software/recsys/setup.m')
load('~/result/tkde_dcf/dmf_results(new_ndcg).mat')
paras = cellfun(@(x) x{3}, result, 'UniformOutput', false);
para = paras{2}; para = cell2struct(para(2:2:end), para(1:2:end),1);
clear('result');

ll = 6;
file_name = '~/result/tkde_dcf/two_stage_item_size.mat';
data2 = readContent('~/data/amazon/ratings_mapped.csv', 'sep', ',');
sz = 2:2:20;
max_iter = 5;
result = cell(ll, max_iter+2);
time = zeros(ll, max_iter+1);
for s = ll:-1:1
    fprintf('%d\n',sz(s));
    data = trim_data(data2, sz(s));
    
    [train, test] = split_matrix(data, 'un', 0.8);
    [P,Q] = dmf(train, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'init', true);
    [B,D] = dmf(train, 'K', 64, 'max_iter', 20, 'rho', para.rho, 'alpha', para.alpha, 'beta', para.beta);
    [M, N] = size(train);
    if M > 20000
        idx = randi(M, 20000, 1);
        train1 = train(idx,:); test1 = test(idx,:);
        P1 = P(idx,:); B1 = B(idx,:);
    else
        train1 = train; test1 = test;
        P1 = P; B1 = B;
    end
    
    for iter=1:max_iter
        [result{s, iter}, time(s, iter)] = mip_search(struct('real', P1, 'code', B1), struct('real', Q, 'code', D), train1, test1, 1000);    
    end
    [result{s, iter+1}, time(s, iter+1)] = mip_search(P1, Q, train1, test1, 200);
    result{s, iter+2} = size(data, 2);
    
    save(file_name, 'result', 'time');
end


