function [B, D] = pph(R, varargin )
%Preference Preserving Hashing for Efficient Recommendation
%   
[opt.lambda, max_iter, k, test, debug] = process_options(varargin, 'lambda', 0.01, 'max_iter', 10, 'K', 20, 'test', [], 'debug',true);
print_info();
[m,n] = size(R);
Rt = R';
rng(200);
B = randn(m,k)*0.1; D = randn(n,k)*0.1;
opt.rmax = max(max(R));
converge = false;
it = 1;
loss0 = 0;
while ~converge
    B = optimize_(Rt, D, B, opt);
    D = optimize_(R, B, D, opt);
    loss = loss_();
    if debug
        fprintf('Iteration=%3d of all optimization, loss=%.1f,', it, loss);
        if ~isempty(test)
            metric = evaluate_rating(test, B, D, 10);
            fprintf('ndcg@1=%.3f', metric.rating_ndcg(1));
        end
        fprintf('\n')
    end
    if it >= max_iter || abs(loss0-loss)<1e-6 * loss || abs(loss0-loss)<1
        converge = true;
    end
    it = it + 1;
    loss0 = loss;
end
[B,D] = rounding(B,D);
function print_info()
    fprintf('pph (K=%d, max_iter=%d, lambda=%f)\n', k, max_iter, opt.lambda);
end
function v = loss_()
    v = 0;
    for u=1:m
        r = Rt(:,u);
        idx = r ~= 0;
        r_ = D(idx, :) * B(u,:)';
        r = r(idx);
        v = v + sum((r - opt.rmax/2 - r_).^2);
    end
    v = v + opt.lambda * sum((sum(B.^2, 2) - opt.rmax/2).^2);
    v = v + opt.lambda * sum((sum(D.^2, 2) - opt.rmax/2).^2);
end
end

function B = optimize_(Rt, D, B, opt)
m = size(Rt, 2);
for u=1:m
    r = Rt(:,u);
    b = B(u,:)';
    idx = r~=0;
    Du = D(idx,:);
    ru = full(r(idx)) - opt.rmax/2;
    H = 2 * (Du' * Du);
    g = @(x) 2 * (Du' * (Du * x)) - 2 * (Du' * ru);
    f = @(x) sum((ru - Du * x).^2);
    options = optimoptions('fminunc','Algorithm','trust-region', ...
        'SpecifyObjectiveGradient',true,'HessianFcn','objective','Display','off');
    B(u,:) = fminunc(@(x) fun(x,H,g,f, opt), b, options);
end
end

function [f,g,H] = fun(b, H1, g1, f1, opt)
% d: kx1 vector
    k = length(b);
    v = norm(b)^2 - opt.rmax/2;
    f = opt.lambda * v^2 + f1(b);
    g = 4*opt.lambda* v * b + g1(b);
    H = 4*opt.lambda*(v*eye(k) + 2 * (b * b')) + H1;
end

function [B,D] = rounding(B,D)
m = size(B,1); n = size(D,1);
B1 = 2*(B>0)-1; D1 = 2*(D>0)-1;
v = sum(D.^2, 2); mu = mean(v); sigma = std(v);
idx1 = v < mu - sigma;
idx2 = v > mu + sigma;
D2 = [-ones(n,1),ones(n,1)];
D2(idx1, 2) = -1; D2(idx2, 1) = 1;
B2 = ones(m, 2);
B = [B1,B2]; D = [D1,D2];
end

