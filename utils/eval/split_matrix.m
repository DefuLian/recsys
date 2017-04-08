function [train, test] = split_matrix(mat, mode, ratio)
if strcmp(mode, 'un') 
    [train, test] = normal_split(mat, ratio);
elseif strcmp(mode, 'in') 
    [train, test] = normal_split(mat.', ratio);
    train = train.';
    test = test.';
elseif strcmp(mode, 'en') 
    [train, test] = entry_split(mat, ratio);
elseif strcmp(mode, 'u') 
    [train, test] = item_split(mat.', ratio);
    train = train.';
    test = test.';
elseif strcmp(mode, 'i') 
    [train, test] = item_split(mat, ratio);
else
    error('Unsupported split mode');
end
end

function [train, test] = normal_split(mat, ratio)
[M, N] = size(mat);
matt = mat.';
train_cell = cell(M, 1);
test_cell = cell(M, 1);
for u=1:M
    rows = matt(:,u);
    [J,I,V] = find(rows);
    samples = randsample(length(J), round(ratio * length(J)));
    bit = false(length(J),1);
    bit(samples) = true;
    train_cell{u} = [u * I(bit), J(bit), V(bit)];
    test_cell{u} = [u * I(~bit), J(~bit), V(~bit)];
end
train_index = cell2mat(train_cell);
test_index = cell2mat(test_cell);
train = sparse(train_index(:,1), train_index(:,2), train_index(:,3), M, N);
test = sparse(test_index(:,1), test_index(:,2), test_index(:,3), M, N);
end

function [train_item, test_item] = item_split(mat, ratio)
[~, N] = size(mat);
indi = datasample(1:N, round(N * ratio), 'replace', false);
ind = false(1,N);
ind(indi) = true;
train_item = mat;
train_item(:,~ind) = 0;
test_item = mat;
test_item(:,ind) = 0;
end

function [train, test] = entry_split(mat, ratio)
[M, N] = size(mat);
[I,J,V] = find(mat);
indi = datasample(1:length(V),round(length(V)*ratio), 'replace', false);
ind = false(1,length(V));
ind(indi) = true;
train = sparse(I(ind), J(ind), V(ind), M, N);
test = sparse(I(~ind), J(~ind), V(~ind), M, N);
end
