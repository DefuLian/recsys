%%
max_iter = 100;
seed = 30;
%rng(seed)
k=100;
a = randn(10000,k);
A = a.' * a;
b = randn(k,1);
x_init = +(randn(k,1)>0);
xx = x_init * 2 - 1;
x_init = x_init * 2 - 1;
l00 = x_init'*A*x_init - 2 * b.' *x_init;
tic;[x1,l1] = bqp(x_init, A, b, 'alg','ccd','max_iter',max_iter);toc
tic;[x2,l2] = bqp(x_init, A, b, 'alg','svr');toc
tic;[x3,l3] = bqp(x_init, A, b, 'alg','bcd','blocksize',16,'max_iter',max_iter);toc
tic;[x4,l4] = bqp(x_init, A, b, 'alg','bcd','blocksize',1,'max_iter',max_iter);toc

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
b = randn(10,1);
A = randn(10,10);
A = A'*A;
xx = A\b;
options = optimoptions('fminunc','Algorithm','trust-region','Display','off',...
        'SpecifyObjectiveGradient',true,'HessianFcn','objective','MaxIterations',20);
xstar = fminunc(@(x) fun(x,A,b), zeros(10,1), options);

f = @(x) x'* A*x - 2*b'*x;
f(xx)
f(xstar)
%%
load C:/Users/liand/Desktop/code/dataset/ml100kdata.mat;
load ~/data/ml100kdata.mat
[B,D] = dmf(Traindata, 'K', 100, 'alpha',0,'beta',0);
[B,D] = dmf(+(Traindata>4), 'K', 32, 'alpha',0,'beta',0, 'rho',0 ,'islogit',true,'alg','ccd','max_iter',1,'init',true);
[B1,D1] = dmf(Traindata, 'K', 32, 'alpha',0,'beta',0, 'rho',0 ,'islogit',false,'alg','bcd','max_iter',20,'init',true, 'debug',true, 'test', Testdata);
metric = evaluate_rating(Testdata, B1, D1, 10);
[ndcg,rmse] = rating_metric(Testdata, B1, D1, 10);
metric2 = rating_recommend(@(mat) dmf(mat, 'K', 32, 'alpha',0,'beta',0, 'rho',0 ,'islogit',false,'alg','bcd','max_iter',20,'init',true), Traindata, 'test', Testdata);
[B,D] = pph(Traindata, 'K', 32, 'max_iter',5, 'lambda',0.1, 'test', Testdata);
metric = evaluate_rating(Testdata, B, D, 10);
metric3 = evaluate_rating(Testdata, B, D, 10);
[B,D] = bccf(Traindata, 'K', 32, 'max_iter',10, 'lambda',0.1, 'test', Testdata);
[B,D] = ch(Traindata, 'K', 32, 'max_iter',100);
[B,D] = DCF(Traindata, 'K', 32, 'max_iter',100, 'alpha',0.1, 'beta',0.1);
%%
[summary1, detail1, time1]=rating_recommend(@(mat) dmf(mat, 'K',32, 'alpha',0,'beta',0, 'rho',0, 'init',true), Traindata+Testdata, 'test_ratio',0.2, 'times',5);
[summary, detail, time]=item_recommend(@(mat) dmf(mat, 'K',32, 'alpha',0,'beta',0, 'rho',0, 'init',true), Traindata+Testdata, 'folds',5);

[a1,b1,c1,d1] = hyperp_search(...
    @(varargin) item_recommend(@(mat) dmf(mat, 'K',32, 'rho',0, 'init',true, varargin{:}), Traindata+Testdata, 'test_ratio',0.2), ...
    @(metric) metric.item_recall(1,end), 'alpha', [0.001,0.01], 'beta',[0.01,0.1,1], 'mode','seq');
[a,b,c,d,e,f] = running(@(mat,varargin) dmf(mat, 'K',32, 'rho',0, 'init',true, varargin{:}), ...
    Traindata+Testdata, 'alpha', [0.001,0.01], 'beta',[1,2,3], 'rating', true);


%%
i=50;
aii = A(i,i);
ai = A(i,:);
x = randn(k,1);
bb = b(i) - ai*x + x(i)*aii;
L = [aii,0.00001;0.00001,0];
[~, X, ~] = psd_ip(L,'precision', 6);

Xi = mvnrnd(zeros(2,1), X, 10);
loss = sum((Xi * C) .* Xi, 2);
%%
data = Traindata+Testdata;

d1 = @(mat) dmf(mat, 'K', 64, 'alpha',1e-4,'beta',0, 'rho',0.01 , 'max_iter',20);
d2 = @(mat) DCF(mat, 'K', 64, 'alpha',1e-4,'beta',1e-4, 'max_iter',20);
d3 = @(mat) dmf(mat, 'K', 64, 'alpha',1e-4,'beta',0, 'rho',0.01 , 'max_iter',20, 'alg', 'bcd', 'blocksize',8);
d4 = @(mat) dmf(mat, 'K', 64, 'alpha',1e-4,'beta',0, 'rho',0.01 , 'max_iter',20, 'islogit', true);
d5 = @(mat) dmf(mat, 'K', 64, 'alpha',1e-4,'beta',0, 'rho',0.01 , 'max_iter',20, 'islogit', true, 'alg', 'bcd', 'blocksize',8);

[summary1, detail1, time1]=rating_recommend(d1, data, 'test_ratio',0.2, 'times',1);
[summary2, detail2, time2]=rating_recommend(d2, data, 'test_ratio',0.2, 'times',1);
[summary3, detail3, time3]=rating_recommend(d3, data, 'test_ratio',0.2, 'times',1);
[summary4, detail4, time4]=rating_recommend(d4, data, 'test_ratio',0.2, 'times',1);
[summary5, detail5, time5]=rating_recommend(d5, data, 'test_ratio',0.2, 'times',1);

[B1,D1] = dmf(Traindata, 'K', 32, 'alpha',0.1,'beta',0.1, 'rho',0.1,'alg', 'svr', 'debug',true);


%%
load ~/data/ml10Mdata.mat;
alg = dmf('rho',0.0100,'alpha',1,'beta',0);

