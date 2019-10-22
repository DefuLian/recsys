function [ metric, qtime ] = mip_search(A, B, train, test, k, n_tree, search_k)
train_t = train';
test(train~=0) = 0;
if isnumeric(A) && isnumeric(B)
    if nargin <6
        tic; result = topk_lookup(A, B, train_t, k, true); qtime = toc;
    else
        %A = normalize(A, 2, 'norm');
        A = NormalizeFea(A);
        A = [A,zeros(size(A,1),1)];
        max_norm = max(sqrt(sum(B.^2, 2)));
        B = B / max_norm;
        item_norm = sum(B.^2, 2);
        B = [B, sqrt(1-item_norm)];
        if nargin == 6
            tic; result = annoy_mips(A', B', train_t, k, n_tree, -1); qtime = toc;
        else
            tic; result = annoy_mips(A', B', train_t, k, n_tree, search_k); qtime = toc;
        end
    end
end

if isstruct(A) && isstruct(B)
    tic; result = mips_exhaustive_struct(A, B, train_t, k);qtime = toc;
end
user_count = sum(test~=0, 2);
idx = user_count > 0.0001; 
test = test(idx,:); result = result(idx,:);

[M,N] = size(test);
cand_count = N - sum(train_t(:,idx)~=0);
[user, rank, item] = find(result);
rank_mat = sparse(user, double(item), rank, M, N);
rank_mat = rank_mat .* (test~=0);
if isexplict(test)
    metric = compute_rating_metric(test, rank_mat, cand_count, min(k,200));
else
    metric = compute_item_metric(test, rank_mat, cand_count, min(k,200));
end
end

function result = mips_exhaustive_struct(A, B, train_t, k)
if isfield(A, 'real') && isfield(A, 'code') && isfield(A, 'word')
    % product quantization
    P = A.real; Q = B.real;
    X = uint32(A.code); Y = uint32(B.code);
    U = A.word; V = B.word;
    M = size(X,1);
    %N = size(Y,1);
    [C,K] = size(U);
    F = size(X,2);
    D = K / F;
    lookuptable = zeros(C, C*F);
    for f=1:F
        idx = (1:D) + (f-1)*D;
        subU = U(:,idx);
        subV = V(:,idx);
        lookuptable(:, (1:C) + (f-1)*C) = subV * subU';
    end
    result = pq_search(X, Y, lookuptable, train_t, k);
    %result = zeros(M, k);
    %for i=1:M
    %    pred = zeros(N,1);
    %    for f=1:F
    %        pred = pred + lookuptable(Y(:, f), (f - 1) * C + X(i, f));
    %    end
    %    [~, result(i,:)] = maxk(pred, k);
    %end
    
elseif isfield(A, 'real') && isfield(A, 'code') % hamming distance ranking
    P = A.real; Q = B.real;
    X = compactbit(A.code > 0); Y = compactbit(B.code > 0);
    M = size(X,1);
    result = topk_lookup(X, Y, train_t, k, false);
elseif isfield(A, 'real') && isfield(A, 'query')
    P = A.real; user = A.query;
    Q = B.real; code = uint32(B.code); center = B.word;
    M = size(P,1);
    code = code - min(code) + 1;
    result = apq_search(user, code, center, train_t, k);
else
    error('unsupported type of inputs');
end

for i=1:M
    items = result(i,:);
    pred = Q(items,:) * P(i, :)';
    %[~,idx] = sort(pred, 'descend');
    [~,idx] = maxk(pred, min(200,k));
    result(i,1:min(200,k)) = items(idx);
end
result = result(:,1:min(200,k));
end
