%mex -largeArrayDims Optimize_DC.cpp
[train, test] = readData('E:/data/backup/implicit_feedback/checkin', 1);
[M,N] = size(train);
basic = readContent('E:/data/backup/implicit_feedback/checkin/basic.txt', 'nrows', M);
category = readContent('E:/data/backup/implicit_feedback/checkin/category.txt', 'nrows', N);
K = 50;           
R = train>0;
W = R * 30;
max_iter = 20;
output_wobias0 = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'item_bias', false, 'user_bias', false);
%[~, ~, ~, ~, metric_feature_d1, used_d1] = CWALS_B(train, basic, category, k, test, 'max_iter',max_iter, 'reg_u', 100, 'reg_i', 100, 'alpha', 30);
%% evaluation of weighting scheme.
pop = sum(R).';
pop = pop ./ sum(pop);

alpha = 0.05;
pop = pop.^alpha ./ max(pop.^alpha);
%[a, d] = NMF_1d(train);
usrw = sum(R, 2);
usrw = usrw ./ sum(usrw);
alpha = 0.2;
usrw = usrw .^alpha ./ max(usrw.^alpha);
for iter =1:5
output_bias_two = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'reg_i', 100, 'reg_u', 300, 'item_w', pop, 'usr_w', usrw);
output_bias_item = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'reg_i', 100, 'reg_u', 300, 'item_w', pop, 'usr_w', ones(M,1));
output_bias_usr = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'reg_i', 100, 'reg_u', 300, 'item_w', ones(N,1), 'usr_w', usrw);
output_bias_none = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'reg_i', 100, 'reg_u', 300, 'item_w', ones(N,1), 'usr_w', ones(M,1));
recall = [recall;[output_bias_item.recall; output_bias_usr.recall;output_bias_two.recall;output_bias_none.recall]];
prec = [prec;[output_bias_item.prec; output_bias_usr.prec;output_bias_two.prec;output_bias_none.prec]];
end

ii = 1:4:24;
prec_item =[mean(prec(ii,[50,100,150,200]));std(prec(ii,[50,100,150,200]))] ;
prec_usr = [mean(prec(ii+1,[50,100,150,200]));std(prec(ii+1,[50,100,150,200]))];
prec_both = [mean(prec(ii+2,[50,100,150,200]));std(prec(ii+2,[50,100,150,200]))];
prec_none = [mean(prec(ii+3,[50,100,150,200]));std(prec(ii+3,[50,100,150,200]))];
prec_vv = [prec_both; prec_item; prec_usr;prec_none];

recall_item =[mean(recall(ii,[50,100,150,200]));std(recall(ii,[50,100,150,200]))] ;
recall_usr = [mean(recall(ii+1,[50,100,150,200]));std(recall(ii+1,[50,100,150,200]))];
recall_both = [mean(recall(ii+2,[50,100,150,200]));std(recall(ii+2,[50,100,150,200]))];
recall_none = [mean(recall(ii+3,[50,100,150,200]));std(recall(ii+3,[50,100,150,200]))];
recall_vv = [recall_both;recall_item; recall_usr; recall_none];
save(sprintf(result_dir, 'weighting_cv.mat'),'prec','recall');

%% evaluate bias effect
result_dir = 'F:/百度云同步盘/onedrive/mypaper/content-aware location recommendation/result/%s';
max_iter = 30;
output_wobias1 = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'item_bias', false, 'user_bias', false, 'method', 'CD');
output_ibias = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'item_bias', true, 'user_bias', false, 'method', 'CD');
output_ubias = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'item_bias', false, 'user_bias', true, 'method', 'CD');
output_uibias_cd = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'item_bias', true, 'user_bias', true, 'method', 'CD');
save(sprintf(result_dir, 'bias_effect_cd.mat'),'output_wobias','output_ibias', 'output_ubias','output_uibias');
%%
output_wobias = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'item_bias', false, 'user_bias', false, 'method', 'ALS');
output_ibias = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'item_bias', true, 'user_bias', false, 'method', 'ALS');
output_ubias = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'item_bias', false, 'user_bias', true, 'method', 'ALS');
output_uibias_als = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', max_iter, 'item_bias', true, 'user_bias', true, 'method', 'ALS');
save(sprintf(result_dir, 'bias_effect_als.mat'),'output_wobias','output_ibias', 'output_ubias','output_uibias');

%%
output_wobias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',100, 'item_bias', false, 'user_bias', false, 'method', 'ALS');
output_ibias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',100, 'item_bias', true, 'user_bias', false, 'method', 'ALS');
output_ubias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',100, 'item_bias', false, 'user_bias', true, 'method', 'ALS');
output_uibias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',100, 'item_bias', true, 'user_bias', true, 'method', 'ALS');
save(sprintf(result_dir, 'feature_bias_effect_als_100i.mat'),'output_wobias','output_ibias', 'output_ubias','output_uibias');

output_wobias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',80, 'item_bias', false, 'user_bias', false, 'method', 'ALS');
output_ibias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',80, 'item_bias', true, 'user_bias', false, 'method', 'ALS');
output_ubias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',80, 'item_bias', false, 'user_bias', true, 'method', 'ALS');
output_uibias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',80, 'item_bias', true, 'user_bias', true, 'method', 'ALS');
save(sprintf(result_dir, 'feature_bias_effect_als_80i.mat'),'output_wobias','output_ibias', 'output_ubias','output_uibias');

output_wobias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',60, 'item_bias', false, 'user_bias', false, 'method', 'ALS');
output_ibias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',60, 'item_bias', true, 'user_bias', false, 'method', 'ALS');
output_ubias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',60, 'item_bias', false, 'user_bias', true, 'method', 'ALS');
output_uibias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',60, 'item_bias', true, 'user_bias', true, 'method', 'ALS');
save(sprintf(result_dir, 'feature_bias_effect_als_60i.mat'),'output_wobias','output_ibias', 'output_ubias','output_uibias');

load(sprintf(result_dir, 'bias_effect_als.mat'))
output_wobias.recall(200)
ind = 1:100;
plot(ind, output_wobias.recall(ind), ind, output_ibias.recall(ind), ind, output_ubias.recall(ind), ind, output_uibias.recall(ind));
%%
%output1 = CWALS_B(W, R, zeros(M,0), zeros(N,0), K, 'test', test, 'max_iter', 10, 'reg_i', 100, 'reg_u', 100, 'item_bias', false, 'user_bias', false);
output_bias = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', 10, 'reg_i', 100, 'reg_u', 350, 'item_bias', false, 'user_bias', false);
[P, Q, U, V, metric] = CWALS(W, R, zeros(M,0), zeros(N,0), K, test, 'max_iter', 10, 'reg_u', 0.01, 'reg_i', 0.01);
%% evaluate efficiency
allm = train + test;
[I, J, Val] = find(allm);
kfold_index = crossvalind('Kfold', nnz(allm), 10);
time_cd = zeros(10,1);
time_als = zeros(10,1);
for f = 1:10
    index = kfold_index <= f;
    train_new = sparse(I(index), J(index), Val(index), M, N);
    R_new = +(train_new > 0);
    W_new = R_new * 30;
    %tic
    %CWALS_B(W_new, R_new, zeros(M,0), zeros(N,0), K, 'max_iter', max_iter, 'item_bias', false, 'user_bias', false,'method', 'CD');
    %time_cd(f) = toc;
    tic
    CWALS_B(W_new, R_new, zeros(M,0), zeros(N,0), K, 'max_iter', max_iter, 'item_bias', false, 'user_bias', false, 'method', 'CD');
    time_cd(f) = toc;
end
save(sprintf(result_dir, 'time_data.mat'),'time_cd','time_als');

R_new = +(allm>0);
W_new = R_new * 30;
kk = 1;
for K =150:50:500
    tic
    CWALS_B(W_new, R_new, zeros(M,0), zeros(N,0), K, 'max_iter', max_iter, 'item_bias', false, 'user_bias', false, 'method', 'CD');
    time_cd(kk) = toc;
    tic
    CWALS_B(W_new, R_new, zeros(M,0), zeros(N,0), K, 'max_iter', max_iter, 'item_bias', false, 'user_bias', false, 'method', 'ALS');
    time_als(kk) = toc;
    kk = kk + 1;
end

%% evaluate convergence
K = 50; max_iter = 100;
output_als_300 = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',80, 'method', 'ALS', 'k-v', 1, 'pos_eval',50);
output_cd_300 = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 300, 'reg_i',80, 'method', 'CD', 'k-v', 1, 'pos_eval',50);
output_als_350 = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 350, 'reg_i',80, 'method', 'ALS', 'k-v', 1, 'pos_eval',50);
output_cd_350 = CWALS_B(W, R, basic, category, K, 'test', test, 'max_iter', max_iter, 'reg_u', 350, 'reg_i',80, 'method', 'CD', 'k-v', 1, 'pos_eval',50);

%%