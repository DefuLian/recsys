function [P,Q,X,Y,U,V] = qcf_init_opq(R,varargin)
[P,Q,~] = process_options(varargin, 'P0', [], 'Q0',[]);
fprintf('qcf_init_opq\n');
if isempty(P) || isempty(Q)
    [P, Q] = iccf(R, varargin{:});
    P = P(:, 1:end-2); Q = Q(:,1:end-2);
end
[P,Q,X,Y,U,V]=train_opq(P, Q, 50);
end

function [P_, Q_, X, Y, U, V] = train_opq(P, Q, num_iter)

k = 256; % fixed number of centers per subspaces (8 bits per subspaces)
d = 8;
dim = size(P, 2);
M = dim/d;

[P_, U, X] = train_pq(P, 5000);
[Q_, V, Y] = train_pq(Q, 5000);
inner_iter = 5000;
for iter = 2: num_iter/2
    R = proj_stiefel_manifold(P'*P_ + Q'*Q_);
    P_proj = P * R;
    Q_proj = Q * R;
    distortion = zeros(2,1);
    for m = 1:M
        idx_space = (1:d) + (m-1)*d;
        
        Psub = P_proj(:, idx_space);

        opts = statset('Display','off','MaxIter',inner_iter);
        [~, centers] = kmeans(Psub, k, 'Options', opts, 'Start', U(:,idx_space), 'EmptyAction', 'singleton');
        %[~, centers] = litekmeans(Psub, k, 'Start', U(:,idx_space), 'MaxIter',inner_iter);
        U(:,idx_space) = centers;
        
        dist = sqdist(centers', Psub');
        [dist, idx] = min(dist);
        X(:,m) = idx;
        
        P_(:,idx_space) = centers(idx,:);
        
        dist = mean(dist);
        distortion(1) = distortion(1) + dist;
        
        
        
        Qsub = Q_proj(:, idx_space);
        
        opts = statset('Display','off','MaxIter',inner_iter);
        [~, centers] = kmeans(Qsub, k, 'Options', opts, 'Start', V(:,idx_space), 'EmptyAction', 'singleton');
        %[~, centers] = litekmeans(Qsub, k, 'Start', V(:,idx_space), 'MaxIter',inner_iter);
        V(:,idx_space) = centers;
        
        dist = sqdist(centers', Qsub');
        [dist, idx] = min(dist);
        Y(:,m) = idx;
        
        Q_(:,idx_space) = centers(idx,:);
        
        dist = mean(dist);
        distortion(2) = distortion(2) + dist;

    end
    fprintf('iter=%d, P distortion : %e, Q distortion : %e\n', iter, distortion(1), distortion(2));
end

end


function [X, centers_table, idx_table, distortion] = train_pq(X, num_iter)

% X: [nSamples, dim] training samples
% M: number of subspacs

k = 256; % fixed number of centers per subspaces (8 bits per subspaces)
d = 8;
dim = size(X, 2);
M = dim/d;


centers_table = zeros(k, dim);
idx_table = zeros(size(X, 1), M);

distortion = 0;

for m = 1:M
    %fprintf('subspace #: %d', m);
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
    
    %fprintf('    distortion in this subspace: %e\n', dist);
    
    X(:,(1:d) + (m-1)*d) = centers(idx,:);
end

end

