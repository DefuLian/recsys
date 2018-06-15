data = readContent('/home/dlian/data/checkin/Gowalla/data.txt');
[M, N] = size(data);
alpha = 100;
K = 100;
[train, test] = split_matrix(data, 'un', 0.8);
%metric = item_recommend(@(mat) , +(train>0), 'test', test);

metric10 = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 10), +(train>0), 'test', test);
metric20 = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 20), +(train>0), 'test', test);
metric30 = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 30), +(train>0), 'test', test);


[train, test] = readData('/home/dlian/data/checkin/Beijing/',1);
[~, ~, ~, ~, metric] = piccf(+(train>0), 'alpha', 30, 'test', test, 'pos_eval', 50);

train_neg = sample_negative(+(train>0)');
mean(sum(train_neg,2))

[P, Q] = splicf(train_neg, 'test', test);

