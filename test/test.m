% this script is used for testing
data = train + test;
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
%% testing tuple-based evalaution
%% testing cold-start issues
%% testing heldout evaluation
%% testing cv evalaution