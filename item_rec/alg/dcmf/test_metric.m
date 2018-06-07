%% yelp
r = 8;
k = 10;
[B00,D00,U0,V0,P0,Q0] = DCMFinit(train, zeros(size(train,1),0), feature, r, 'maxItr',10,'eta',[0.01 10000], 'debug', 1, 'test', test);
B01 = sign(B00); B01(B01 == 0) = 1;
D01 = sign(D00); D01(D01 == 0) = 1;
[BB,DD,BB1,DD1] = DCMF(train, zeros(size(train,1),0), feature, r, 'maxItr',10,'maxItr2',5,'eta',[0 2], 'debug', 1,...
    'B0',B01, 'D0',D01, 'U0',U0, 'V0',V0, 'P0',P0, 'Q0',Q0, 'test', test);
ndcg_DCMF_opt = rating_metric(test, BB, DD, k);
ndcg_DCMF_ini = rating_metric(test, BB1, DD1, k);

%% amazon

r = 8;
k = 10;
tic
[B00,D00,U0,V0,P0,Q0] = DCMFinit(train, zeros(size(train,1),0), feature, r, 'maxItr',10,'eta',[0.01 500000], 'debug', 1, 'test', test);
toc
B01 = sign(B00); B01(B01 == 0) = 1;
D01 = sign(D00); D01(D01 == 0) = 1;
tic
[BB,DD,BB1,DD1] = DCMF(train, zeros(size(train,1),0), feature, r, 'maxItr',10,'maxItr2',1,'eta',[0 5], 'debug', 1,...
    'B0',B01, 'D0',D01, 'U0',U0, 'V0',V0, 'P0',P0, 'Q0',Q0, 'test', test);
toc
ndcg_DCMF_opt = rating_metric(test, BB, DD, k);
ndcg_DCMF_ini = rating_metric(test, BB1, DD1, k);
%% amazon fast

r = 8;
k = 10;
tic;
[B00,D00,U0,V0,P0,Q0] = DCMFinit_fast(train, zeros(size(train,1),0), feature, r, 'maxItr',10,'eta',[0.01 500000], 'debug', 1, 'test', test);
toc;
B01 = sign(B00); B01(B01 == 0) = 1;
D01 = sign(D00); D01(D01 == 0) = 1;
tic
[BB,DD,BB1,DD1] = DCMF_fast(train, zeros(size(train,1),0), feature, r, 'maxItr',10,'maxItr2',1,'eta',[0 5], 'debug', 1,...
    'B0',B01, 'D0',D01, 'U0',U0, 'V0',V0, 'P0',P0, 'Q0',Q0, 'test', test);
toc;
ndcg_DCMF_opt = rating_metric(test, BB, DD, k);
ndcg_DCMF_ini = rating_metric(test, BB1, DD1, k);

%% movie


[B,D,B1,D1] = DCMF(train,  zeros(size(train,1),0), feature, r, 'maxItr',10, 'alpha',[0.1 0.1],'eta',[0 10], 'debug', 1);
ndcg_DCMF_16b_a_10_e10_f18 = rating_metric(test,B,D,k);
ndcg_DCMFinit_16b_a10_e10_f18 = rating_metric(test,B1,D1,k); 

option.maxItr = 10;
[B,D,B1,D1] = DCF(train, r, 0.1, 0.1, option);
ndcg_DCF = rating_metric(test,B',D',k);



ndcg_DCFinit_8b_1 = rating_metric(test,B1',D1',k);


% the number of feature
r = 8;
l = [7000,5000,2500,1000,500,100,50,25,10];
ndcg_DCMF_8b_f0 = zeros(1,9);
ndcg_DCMFinit_8b_f0 = zeros(1,9);
for i = 1:9
    len = l(i);
    feature(:,len+1:size(feature,2)) = [];
    [B, D, B1, D1] = DCMF(train,  zeros(size(train,1),0), feature, r, 'maxItr',10,'maxItr2',5, 'alpha',[0.1 0.1],'eta',[0 1], 'debug', 1);
    ndcg = rating_metric(test, B, D, k);
    ndcg_DCMF_8b_f0(i) = ndcg(10);
    ndcg = rating_metric(test, B1, D1, k);
    ndcg_DCMFinit_8b_f0(i) = ndcg(10);
end



% single init
r=8;
k=10;
option.maxItr = 10;
option.debug = true;
option.maxItr2 = 1;

[B0,D0,X0,Y0] = DCFinit(train, r, 0.1, 0.1, option);
ndcg_dcf = rating_metric(test, B0', D0', k);
B0 = sign(B0); B0(B0 == 0) = 1;
D0 = sign(D0); D0(D0 == 0) = 1;
option.B0 = B0;
option.D0 = D0;
option.X0 = X0;
option.Y0 = Y0;
[B,D,B1,D1] = DCF(train, r, 0.1, 0.1, option);
ndcg_DCF_8b = rating_metric(test, B', D', k);
ndcg_DCFinit_8b = rating_metric(test, B1', D1', k);

[B00,D00,U0,V0,P0,Q0] = DCMFinit(train,zeros(size(train,1),0), feature, r, 'maxItr',10, 'alpha',[0.1 0.1],'lambda',[0 10], 'debug', 0);
B00 = sign(B00); B00(B00 == 0) = 1;
D00 = sign(D00); D00(D00 == 0) = 1;

[BB,DD,BB1,DD1] = DCMF(train,  zeros(size(train,1),0), feature, r, 'maxItr',10,'maxItr2',5, 'alpha',[0.1 0.1],'eta',[0 0.01], 'debug', 0,...
    'B0',B0', 'D0',D0', 'U0',U0, 'V0',V0, 'P0',P0, 'Q0',Q0);
ndcg_DCMF_16b_e0_001 = rating_metric(test, BB, DD, k);
ndcg_DCMFinit_16b_e0_001 = rating_metric(test, BB1, DD1, k);


