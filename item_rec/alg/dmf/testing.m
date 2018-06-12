%%
max_iter = 100;
seed = 30;
rng(seed)
a = randn(10000,100);
A = a.' * a;
b = randn(100,1);
x_init = +(randn(100,1)>0);
xx = x_init * 2 - 1;
x_init = x_init * 2 - 1;
l00 = x_init'*A*x_init - 2 * b.' *x_init;
[x1,l1] = bqp(x_init, A, b, 'alg','ccd','max_iter',max_iter);
[x2,l2] = bqp(x_init, A, b, 'alg','svr');
[x3,l3] = bqp(x_init, A, b, 'alg','bcd','block_size',50,'max_iter',max_iter);

rng(seed)
a = randn(10000,100);
A = a.' * a;
b = randn(100,1);
x_init = +(randn(100,1)>0);
xx = x_init * 2 - 1;
x_init = x_init * 2 - 1;
options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
xx_init = quadprog(A, -b, [], [], [], [], -ones(100,1), ones(100,1), [], options);
%l00 = xx_init'*A*xx_init - 2 * b.' *xx_init;
xx_init = +(xx_init>0) * 2 - 1;
l01 = xx_init'*A*xx_init - 2 * b.' *xx_init;
[x4,l4] = bqp(xx_init, A, b, 'alg','ccd','max_iter',max_iter);
[x5,l5] = bqp(xx_init, A, b, 'alg','svr');
[x6,l6] = bqp(xx_init, A, b, 'alg','bcd','block_size',50,'max_iter',max_iter);
%[x3,l3] = bqp(x_init, A, b, 'alg','mix','max_iter',1);
%c = zeros(100,1);
%DCDmex(xx, A, b, c, 1);
%sum(x ~=xx)

%%

[B,D] = dmf(Traindata, 'K', 100, 'alpha',0,'beta',0);
[B,D] = dmf(+(Traindata>4), 'K', 32, 'alpha',0,'beta',0, 'rho',0 ,'islogit',true,'alg','ccd','max_iter',10);