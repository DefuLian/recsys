function mat_folds = kFolds(mat, folds, mode)
% mode of cross validation: 
%   un: user side normal 
%   in: item side normal
%   en: entry size normal
%   u: user
%   i: item
if strcmp(mode, 'un') 
    mat_folds = kNormalFolds(mat, folds);
elseif strcmp(mode, 'in') 
    mat_folds = kNormalFolds(mat.', folds);
    for k=1:folds
        mat_folds{k} = mat_folds{k}.';
    end
elseif strcmp(mode, 'en') 
    mat_folds = kEntryFolds(mat, folds);
elseif strcmp(mode, 'u') 
    mat_folds = kItemFolds(mat.', folds);
    for k=1:folds
        mat_folds{k} = mat_folds{k}.';
    end
elseif strcmp(mode, 'i') 
    mat_folds = kItemFolds(mat, folds);
else
    error('Unsupported mode of cross validation ');
end
end

function mat_folds = kEntryFolds( mat, folds )
mat_folds = cell(folds,1);
[M, N] = size(mat);
[I, J, V] = find(mat);
index = crossvalind('Kfold', length(J), folds);
for k = 1:folds
    mat_folds{k} = [I(index==k),J(index==k),V(index==k)];
end
for k = 1:folds
    mat_folds{k} = sparse(mat_folds{k}(:,1), mat_folds{k}(:,2), mat_folds{k}(:,3), M, N);
end
end

function mat_folds = kNormalFolds( mat, folds )
%SplitMatrix Split matrix based on the given ratio
matt = mat.';
mat_folds = cell(folds,1);
[M, N] = size(mat);
tuple_cell = cell(M, folds);
for u=1:M
    rows = matt(:,u);
    [J, I, V] = find(rows);
    index = crossvalind('Kfold', length(J), folds);
    for k = 1:folds
        tuple_cell{u, k} = [u*I(index==k),J(index==k),V(index==k)];
    end
end
for k = 1:folds
    smat_fold = cell2mat(tuple_cell(:,k));
    mat_folds{k} = sparse(smat_fold(:,1), smat_fold(:,2), smat_fold(:,3), M, N);
end
end

function mat_folds = kItemFolds(mat, folds)
mat_folds = cell(folds,1);
[~, N] = size(mat);
index = crossvalind('Kfold', N, folds);
for k=1:folds
    sub_mat = mat;
    sub_mat(:,index ~= k) = 0;
    mat_folds{k} = sub_mat;
end
end