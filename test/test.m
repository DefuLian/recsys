% this script is used for testing

[train, test] = readData('/home/dlian/data/subcheckin', 1);

data = train+test;
R = data>0;
metric1 = item_recommend(@iccf, train>0, 'test', test, 'topk', 200);
metric2 = item_recommend(@iccf, R,'split_ratio',0.8,'split_mode','en','topk', 200, 'times',3);
metric3 = item_recommend(@iccf, R, 'folds', 5, 'topk', 200, 'fold_mode', 'en');
[~, out] = item_recommend(@iccf, train>0, 'topk', 200);

%output_wobias0 = iccf(R, 'K',50, 'max_iter', max_iter);

%% testing split_matrix

[train, test] = split_matrix(data,'un', .8);
assert(nnz((train+test) ~= data)==0)
assert(nnz(train.*test)==0)
[train, test] = split_matrix(data,'in', .8);
assert(nnz((train+test) ~= data)==0)
assert(nnz(train.*test)==0)
[train, test] = split_matrix(data,'u', .8);
assert(nnz((train+test) ~= data)==0)
assert(nnz(train.*test)==0)
[train, test] = split_matrix(data,'i', .8);
assert(nnz((train+test) ~= data)==0)
assert(nnz(train.*test)==0)
[train, test] = split_matrix(data,'en', .8);
assert(nnz((train+test) ~= data)==0)
assert(nnz(train.*test)==0)

%% test k-folds
folds = kFolds(data, 5, 'un');
final = folds{1};
for i=2:length(folds)
    final = final + folds{i};
end
assert(nnz(final~=data)==0);
folds = kFolds(data, 5, 'in');
final = folds{1};
for i=2:length(folds)
    final = final + folds{i};
end
assert(nnz(final~=data)==0);
folds = kFolds(data, 5, 'en');
final = folds{1};
for i=2:length(folds)
    final = final + folds{i};
end
assert(nnz(final~=data)==0);
folds = kFolds(data, 5, 'i');
final = folds{1};
for i=2:length(folds)
    final = final + folds{i};
end
assert(nnz(final~=data)==0);
folds = kFolds(data, 5, 'u');
final = folds{1};
for i=2:length(folds)
    final = final + folds{i};
end
assert(nnz(final~=data)==0);
%% testing topk ranking and sorting
metric1 = item_recommend(@iccf, train>0, 'test', test, 'topk', 200, 'max_iter', 10);
metric = item_recommend(@iccf, train>0, 'test', test, 'max_iter', 10);
assert(nnz(metric1.recall~=metric.recall)==0)
assert(nnz(metric1.prec~=metric.prec)==0)
assert(nnz(metric1.ndcg~=metric.ndcg)==0)
assert(nnz(metric1.map~=metric.map)==0)
%% testing topk item generating
metric1 = item_recommend(@iccf, train>0, 'test', test, 'topk', 200);
[~,topmat] = item_recommend(@iccf, train>0,'topk',200);
[M, N] = size(train);
mat_rank = sparse(topmat(:,1),topmat(:,2),topmat(:,3), M, N);
mat_rank = mat_rank .* (test>0);
user_count = sum(test~=0 & xor(train~=0, test~=0),2);
cand_count = N - sum(train,2);
metric = compute_item_metric(mat_rank, user_count, cand_count, 200, 200);
assert(nnz(metric1.recall(1,:)~=metric.recall)==0)
assert(nnz(metric1.prec(1,:)~=metric.prec)==0)
assert(nnz(metric1.ndcg(1,:)~=metric.ndcg)==0)
assert(nnz(metric1.map(1,:)~=metric.map)==0)
%% testing tuple-based evalaution
[M,N] = size(test);
test_test = test>0;
user_ind = false(M,1);
num_samples = 2000;
users = randsample(M,num_samples);
user_ind(users) = true;
test_test(~user_ind,:) = 0;
test_test_test = full(test_test(user_ind,:)).';
users = find(user_ind);
[J,I] = ind2sub([N,num_samples], 1:(num_samples*N));
V = test_test_test(1:(num_samples*N));
I = users(I).';
metric  = item_recommend(@iccf, train>0, 'test', test_test, 'topk', 200);
metric1 = item_recommend(@iccf, train>0, 'test', [I',J',V'], 'topk', 200);
assert(nnz(metric1.recall~=metric.recall)==0)
assert(nnz(metric1.prec~=metric.prec)==0)
assert(nnz(metric1.ndcg~=metric.ndcg)==0)
assert(nnz(metric1.map~=metric.map)==0)
%% testing cold-start issues
metric1 = item_recommend(@iccf, R, 'folds', 5, 'topk', 200, 'fold_mode', 'u');
metric2 = item_recommend(@iccf, R, 'folds', 5, 'topk', 200, 'fold_mode', 'i');
metric3 = item_recommend(@iccf, R, 'split_ratio', 0.8, 'topk', 200, 'split_mode', 'u');
metric4 = item_recommend(@iccf, R, 'split_ratio', 0.8, 'topk', 200, 'split_mode', 'i');

%% testing cutoff and topk
metric0 = item_recommend(@iccf, train>0, 'test', test);
metric1 = item_recommend(@iccf, train>0, 'test', test, 'topk', 150);
metric2 = item_recommend(@iccf, train>0, 'test', test, 'cutoff', 50);
metric3 = item_recommend(@iccf, train>0, 'test', test, 'topk', 50,'cutoff',50);
metric4 = item_recommend(@iccf, train>0, 'test', test, 'topk', 150,'cutoff',50);
metric5 = item_recommend(@iccf, train>0, 'test', test, 'topk', 50,'cutoff',150);
metric6 = item_recommend(@iccf, train>0, 'test', test, 'topk', 150,'cutoff',150);
assert(nnz(metric0.recall(:,1:10) ~= metric1.recall(:,1:10))==0)
assert(nnz(metric0.recall(:,1:10) ~= metric2.recall(:,1:10))==0)
assert(nnz(metric0.recall(:,1:10) ~= metric3.recall(:,1:10))==0)
assert(nnz(metric0.recall(:,1:10) ~= metric4.recall(:,1:10))==0)
assert(nnz(metric0.recall(:,1:10) ~= metric5.recall(:,1:10))==0)
assert(nnz(metric0.recall(:,1:10) ~= metric6.recall(:,1:10))==0)

%%
r = [1, 0, 1, 0, 0, 1, 1, 1];
r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0];
rank = 1:length(r);
r = sparse(r);
rank(r==0)=0;
compute_ndcg(r,rank,length(r))
compute_prec_recall(rank,nnz(r),length(r))

