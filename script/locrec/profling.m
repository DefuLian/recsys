metric10 = item_recommend(@geomf, train, 'test', test, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 10);
metric1000 = item_recommend(@geomf, train, 'test', test, 'alpha', alpha, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 1000);
save('/home/dlian/code/matlab/recsys/tmp.mat', 'metric1000', '-append')
metric500 = item_recommend(@geomf, train, 'test', test, 'alpha', alpha, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 500);
save('/home/dlian/code/matlab/recsys/tmp.mat', 'metric500', '-append')
metric100 = item_recommend(@geomf, train, 'test', test, 'alpha', alpha, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 100);
save('/home/dlian/code/matlab/recsys/tmp.mat', 'metric100', '-append')
metric50 = item_recommend(@geomf, train, 'test', test, 'alpha', alpha, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 50);
save('/home/dlian/code/matlab/recsys/tmp.mat', 'metric50', '-append')

metric5 = item_recommend(@geomf, train, 'test', test, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 5);

metric5 = item_recommend(@graph_wals, +(data>0), 'test_ratio', 0.2, 'topk', 100, 'item_sim', item_sim, 'K', 50, 'eta_i', 1, 'max_iter',10);

metric55 = item_recommend(@graph_wals, train, 'test', test, 'topk', 100, 'K', 50, 'eta_i', 1, 'max_iter',10);



metric10_1 = item_recommend(@geomf, train, 'test', test, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 10);


[U, V] = geomf(train, 'Y', item_grid, 'reg_1', 10);
[P,Q,U,V] = iccf(train, 'Y', item_grid, 'K', 300,'reg_i', reg_i);
t = item_grid.' * Q;
F = size(item_grid,2);
mat = item_grid.' * item_grid + spdiags(ones(F,1),0, F, F);
V = mat \ t;
save('/home/dlian/code/matlab/x.mat', 'P', 'Q', 'U', 'V', '-append');
p = P(18,1:300).';
Y = V(:,1:300);


save('/home/dlian/code/matlab/x.mat', 'x1', '-append');

opts.tol = 5e-3;
opts.nonneg = 1;
opts.nonorth = 1; opts.delta = 0.01 ; opts.nu = 0;
opts.rho = 0; x1 = yall1(Y.', p, opts); x1(x1<1e-10) = 0; x1 = sparse(x1); nnz(x1) 

x1 = lassononneg(Y.', p, 0.13);nnz(x1)
[I,~,V] = find(x1);
dlmwrite('/home/dlian/data/checkin/Beijing/test/x1.txt',[I-1,V],'delimiter', '\t', 'precision', '%d');
[I,~,V] = find(x);
dlmwrite('/home/dlian/data/checkin/Beijing/test/x.txt',[I-1,V],'delimiter', '\t', 'precision', '%d');
[I,~,V] = find(y);
dlmwrite('/home/dlian/data/checkin/Beijing/test/y.txt',[I-1,V],'delimiter', '\t', 'precision', '%d');

[I,~,V] = find(train(18,:).');
dlmwrite('/home/dlian/data/checkin/Beijing/test/data.txt',[I-1,V],'delimiter', '\t', 'precision', '%d');

