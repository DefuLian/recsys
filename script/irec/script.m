data = readContent('/home/dlian/data/checkin/Gowalla/data.txt');
[M, N] = size(data);
alpha = 100;
K = 100;
[train, test] = split_matrix(data, 'un', 0.8);
%metric = item_recommend(@(mat) , +(train>0), 'test', test);

metric10 = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 10), +(train>0), 'test', test);
metric20 = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 20), +(train>0), 'test', test);
metric30 = item_recommend(@(mat) piccf(mat, 'alpha', alpha, 'K', K, 'max_iter', 30), +(train>0), 'test', test);


[P, Q] = spicf(+(train>0), 'alpha', alpha, 'K', K, 'max_iter', 10,'test', test, 'reg_u', 50, 'reg_i', 50);

for i=1:10
    load(sprintf('pq%d.mat',i))
    display([max(Q(:)),min(Q(:)), max(P(:)),min(P(:))])
end