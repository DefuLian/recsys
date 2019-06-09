function [B, D] = dmf(R, varargin)
%%% [B,D]=dmf(R, 'K', 64, 'max_iter', 30, 'debug', true, 'loss', 'logit', 'beta', 0.01, 'lambda', 0.01);
%%% optimize 
%%% \sum_{(i,j)\in \Omega} \ell(r_{ij}, b_i'd_j) + 
%%% \rho \sum_{(i,j)\notin \Omega} (b_i'd_j)^2 + 
%%% \alpha(|B-P_b|_F^2 + |D-Q_b|_F^2) + 
%%% \beta (|B-P_d|_F^2 + |D-Q_d|_F^2)
%%% s.t. P_b'1=0, Q_b'1=0, P_d'P_d=I and B_d'B_d=I.
%%% R rating matrix of size m x n
%%% K dimension of hamming space
%%% max_iter the max number of iterations
%%% loss loss function for optimization
%%% rho regularization coefficient for interaction/implicit regularization
%%% alpha regularization coefficient for balanced condition
%%% beta  regularization coefficient for decorrelation condition
[init, opt] = process_options(varargin,'init',false);
if init
    [B,D] = dmf_(R, 'init', true, opt{:});
else
    [B0,D0] = dmf_(R, 'init', true, opt{:});
    %B0 = 2*(B0>0) - 1; D0 = 2*(D0>0) - 1;
    [B,D] = dmf_(R, 'B0', B0, 'D0', D0, 'init', false, opt{:});
end
end

function [B, D] = dmf_(R, varargin)
[k, max_iter, debug, islogit, alpha, beta, rho, alg, bsize, init, B, D, test] = process_options(varargin, 'K', 64, 'max_iter', 10, 'debug', true, ...
    'islogit', false, 'alpha',0.01, 'beta', 0.01, 'rho', 0.01, 'alg', 'ccd','blocksize',32, 'init', false,...
    'B0',[], 'D0',[], 'test',[]);
print_info();
if ~islogit && ~init
    R = scale_matrix(R, k);
end
[m, n]=size(R);
rng(10);
if isempty(B)
    B = randn(m,k)*0.1; 
    if ~init
        B = 2*(B>0)-1; 
    end
end
if isempty(D)
    D = randn(n,k)*0.1;
    if ~init
        D = 2*(D>0)-1;
    end
end
Rt = R.';
opt.rho = rho;
opt.alpha = alpha;
opt.beta = beta;
opt.islogit = islogit;
opt.alg = alg;
opt.bsize = bsize;
opt.init = init;
converge = false;
iter = 1;
%for iter=1:max_iter
while ~converge
    B0 = B; D0 = D;
    P_b = B-repmat(mean(B),m,1); P_d = sqrt(m) * proj_stiefel_manifold(B);
    Q_b = D-repmat(mean(D),n,1); Q_d = sqrt(n) * proj_stiefel_manifold(D);
    DtD = D'*D;
    B = optimize(Rt, D, B, DtD, P_b, P_d, opt);
    BtB = B'*B;
    D = optimize(R,  B, D, BtB, Q_b, Q_d, opt);
    loss = loss_();
    if debug
        fprintf('Iteration=%3d of all optimization, loss=%.1f,', iter-1, loss);
        if ~isempty(test)
            %metric = evaluate_rating(test,B,D,10);
            metric = evaluate_item(R, test, B, D, 200, 200);
            if isexplict(test)
                fprintf('recall@50=%.3f, recall@100=%.3f', metric.item_recall_like(1,50), metric.item_recall_like(1,100));
            else
                fprintf('recall@50=%.3f, recall@100=%.3f', metric.item_recall(1,50), metric.item_recall(1,100));
            end
        end
        fprintf('\n')
    end
    if iter >= max_iter || (norm(B-B0) <1e-3 && norm(D-D0)<1e-3)
        converge = true;
    end
    iter = iter + 1;
end
function print_info()
    if strcmpi(alg, 'ccd')
        alg_name = alg;
    else
        alg_name = sprintf('%s+%d', alg, bsize);
    end
    if init
        if islogit
            fprintf('dmf_logit_init(K=%d, max_iter=%d, rho=%f, alpha=%f, beta=%f, optalg=%s)\n', k, max_iter, rho, alpha, beta, alg_name);
        else
            fprintf('dmf_init(K=%d, max_iter=%d, rho=%f, alpha=%f, beta=%f, optalg=%s)\n', k, max_iter, rho, alpha, beta, alg_name);
        end
    else
        if islogit
            fprintf('dmf_logit(K=%d, max_iter=%d, rho=%f, alpha=%f, beta=%f, optalg=%s)\n', k, max_iter, rho, alpha, beta, alg_name);
        else
            fprintf('dmf(K=%d, max_iter=%d, rho=%f, alpha=%f, beta=%f, optalg=%s)\n', k, max_iter, rho, alpha, beta, alg_name);
        end
    end
end
function val = loss_()
    val = 0;
    for u=1:m
        r = Rt(:,u);
        idx = r ~= 0;
        r_ = D(idx, :) * B(u,:)';
        r = r(idx);
        if opt.islogit
            val = val + sum(logitloss(r .* r_)) - opt.rho * sum(r_.^2);
        else
            val = val + sum((r - r_).^2) - opt.rho * sum(r_.^2);
        end
    end
    val = val + opt.rho*sum(sum((B'*B) .* (D'*D)));
    val = val + opt.alpha * (norm(B-P_b,'fro')^2 + norm(D-Q_b,'fro')^2);
    val = val + opt.beta * (norm(B-P_d,'fro')^2 + norm(D-Q_d,'fro')^2);
end
    
end

function B = optimize(Rt, D, B, DtD, P_b, P_d, opt)
if opt.init
    B = optimize_real(Rt, D, B, DtD, P_b, P_d, opt);
else
    B = optimize_binary(Rt, D, B, DtD, P_b, P_d, opt);
end
end
function B = optimize_real(Rt, D, B, DtD, P_b, P_d, opt)
max_iter = 1;
m = size(Rt, 2);
lambda = @(x) tanh((abs(x)+1e-16)/2)./(abs(x)+1e-16)./4;
X = opt.alpha*P_b + opt.beta*P_d;
for u=1:m
    b = B(u,:);
    r = Rt(:,u);
    idx = r ~= 0;
    Du = D(idx, :);
    r_ = Du * b.';
    %if ~strcmpi(opt.alg,'ccd')
    %    if ~opt.islogit
    %        H = opt.rho * DtD + (1 - opt.rho)*(Du.' * Du) + (opt.alpha+opt.beta+1e-3)*diag(ones(length(b),1));
    %        f = Du.' * r(idx) + X(u,:).';
    %    else
    %        H = opt.rho * DtD + Du.' * diag(lambda(r_) - opt.rho) * Du + (opt.alpha+opt.beta+1e-3)*diag(ones(length(b),1));
    %        f = 1/4 * Du.' * r(idx) + X(u,:).';
    %    end
    %    B(u,:) = H\f;
    %else
        B(u,:) = ccd_logit_mex(full(r(idx)), Du, b,  opt.rho * (DtD - Du'*Du), X(u,:), r_, opt.islogit, max_iter, opt.alpha+opt.beta+1e-3);
    %end
   
end
end
function B = optimize_binary(Rt, D, B, DtD, P_b, P_d, opt)
max_iter = 1;
m = size(Rt, 2);
lambda = @(x) tanh((abs(x)+1e-16)/2)./(abs(x)+1e-16)./4;
X = opt.alpha*P_b + opt.beta*P_d;
for u=1:m
    b = B(u,:)';
    r = Rt(:,u);
    idx = r ~= 0;
    Du = D(idx, :);
    if ~opt.islogit
        H = opt.rho * DtD + (1 - opt.rho) * (Du.' * Du);
        f = Du.' * r(idx) + X(u,:).';
        %B(u,:) = bqp(b.', (H+H')/2, f, 'alg', opt.alg, 'max_iter',max_iter, 'blocksize', opt.bsize);
        B(u,:) = bqp(b, (H+H')/2, f, opt.alg, max_iter, opt.bsize);
        %r_ = Du * b.';
        %B(u,:) = ccd_logit_mex(r(idx), Du, b, opt.rho * (DtD - Du'*Du), X(u,:), r_, opt.islogit, max_iter);
    else
        if ~strcmpi(opt.alg,'ccd')
            r_ = Du * b;
            H = opt.rho * DtD + Du.' * diag(lambda(r_) - opt.rho) * Du;
            f = 1/4 * Du.' * r(idx) + X(u,:).';
            %B(u,:) = bqp(b, (H+H')/2, f, 'alg', opt.alg, 'max_iter',max_iter, 'blocksize', opt.bsize);
            B(u,:) = bqp(b, (H+H')/2, f, opt.alg, max_iter, opt.bsize);
        else
            r_ = Du * b;
            B(u,:) = ccd_logit_mex(full(r(idx)), Du, b, opt.rho * (DtD - Du'*Du), X(u,:), r_, opt.islogit, max_iter);
        end
    end
end
end

function R = scale_matrix(R, s)
maxS = max(max(R));
minS = min(R(R~=0));
[I, J, V] = find(R);
if maxS ~= minS
    VV = (V-minS)/(maxS-minS);
    VV = 2 * s * VV - s + 1e-10;
else
    VV = V .* s ./ maxS;
end
R = sparse(I, J, VV, size(R,1), size(R,2));
end

function v = logitloss(v)
if v>-500
    v = log(1+exp(-v));
else
    v = -v;
end
end