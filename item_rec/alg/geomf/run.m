%
matlabpool ('open',6)

%% read data from files
[train, test] = DataRead('D:/dove/checkin/data/exp');
Y = ContentRead('C:/Users/dane/dove/data/all/item_content.txt', 1);
%%
data = ContentRead('C:/Users/dane/dove/data/all/data.txt', 1);
%[train, test] = DataRead('C:/Users/dane/dove/data/all', 0);
[train, test] = DataRead('C:/Users/dane/dove/data/', 1);
[TR, TE] = SplitMatrix(data, 0.7);
W = GetWeight(train);
%%
load data.mat
load uv.mat
%%
[U1, V1, X] = NN_WALS_Ind(train, content, 50);
%%
tic
[U, V] = WALS(GetWeight(train), train>0, 200 );
%[U, V] = WALS(W, R, 50 );
toc;
%%
mat = Predict(test, R, U, V.');
matk = PredictK(test, train, U, V.',100);
user_count = sum(test>0 & xor(train>0, test>0), 2);
[p,r] = PrecRecall(mat,user_count,100);
[pp,rr] = PrecRecall(matk,user_count, 100);
%%
test(train>0) = 0;
[rank, cand_count ] = Predict(test, train, U, V.');
[AUC, Recall, Precision] = Evaluate( rank, cand_count, 100);
%
%%

%%
[UU, VV] = ALS(train, 50);
%%
tic
X = WNNLS(train, Y, 'epsilon', 1e-10);
toc
tic;
[rank, cand_count ] = Predict(test, train, X, Y.');
[AUC5, Recall5, Precision5] = Evaluate( rank, cand_count, 100);
toc;
%%
%nu = spdiags(1./sum(R, 2), 0, size(R, 1), size(R,1));
%rank = nu * sparse(u, r, 1, size(R,1), numItems);
%rank1 = rank(:,1:100);
%recall = cumsum(rank1,2);
%mr1 = mean(recall);
%plot(1:100, mr1);

%%
matlabpool('close');

%%
Yt = Y.';
YtY = Yt * Y;
W = GetWeight(train);
Wt = W.';
R = train > 0;
Rt = R.';
x = sparse(size(Y,2),1);%x = Xt(:, u);
w = Wt(:, u);
r = Rt(:, u);
Ind = w>0 | r>0; wu = w(Ind); Wu = spdiags(wu, 0, length(wu), length(wu));
ru = r(Ind);
sub_Y = Y(Ind, :);
sub_Yt = Yt(:, Ind);
YCY = sub_Yt * Wu * sub_Y + YtY;
grad_invariant = - sub_Yt * (wu .* ru + ru);
x = LineSearch(YCY, grad_invariant, x);
ul = loc(x>0,:);
vl = full(x(x>0));
vl = [ul,vl];