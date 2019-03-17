function [P, Q, X, Y, U, V] = qcf_init_pq(R, varargin)
[P,Q,~] = process_options(varargin, 'P0', [], 'Q0',[]);
fprintf('qcf_init_pq\n');
if isempty(P) || isempty(Q)
    [P, Q] = iccf(R, varargin{:});
    P = P(:, 1:end-2); Q = Q(:,1:end-2);
end

[P, U, X] = train_pq(P, 5000);
[Q, V, Y] = train_pq(Q, 5000);
end

function [X, centers_table, idx_table, distortion] = train_pq(X, num_iter)

% X: [nSamples, dim] training samples
% M: number of subspacs

k = 256; % fixed number of centers per subspaces (8 bits per subspaces)
d = 8;
dim = size(X, 2);
M = dim/d;

%num_iter = 100;

%centers_table = cell(M, 1);
centers_table = zeros(k, dim);
idx_table = zeros(size(X, 1), M);

distortion = 0;

for m = 1:M
    fprintf('subspace #: %d', m);
    Xsub = X(:, (1:d) + (m-1)*d);
    
    %opts = statset('Display','iter','MaxIter',num_iter);
    opts = statset('Display','off','MaxIter',num_iter);
    [~, centers] = kmeans(Xsub, k, 'Options', opts, 'EmptyAction', 'singleton');
    %[~, centers] = litekmeans(Xsub, k, 'MaxIter', num_iter);
    
    %centers_table{m} = centers;
    centers_table(:,(1:d) + (m-1)*d) = centers;
    
    dist = sqdist(centers', Xsub');
    [dist, idx] = min(dist);
    idx_table(:,m) = idx(:);
    
    % compute distortion
    dist = mean(dist);
    distortion = distortion + dist;
    
    fprintf('    distortion in this subspace: %e\n', dist);
    
    X(:,(1:d) + (m-1)*d) = centers(idx,:);
end

end

