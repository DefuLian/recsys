function [ U, V, X ] = NN_WALS( R, Y, K, varargin )
%Non-negative Weighted Alternative Least Square for POI recommendation with
%activity information, in particular, 
% L = 1/2 * || W .* (R - PQ' - XY') ||_F^2 + reg_u / 2 * ||P||_F^2 + ...
% reg_i /2 * ||Q||_F^2 + reg_1 ||X||_1 , subject to X >= 0
%   R: user-item rating matrix
%   U, V: user and item latent vectors
%   Y: activity vectors for items(POIs)
%   X: activity vectors for users
%   K: dimension of latent vectors
%   num_iter: the maximal iteration
%   tol: tolerance for a relative stopping condition
%   reg_u, reg_i: regularization for users latent vectors and items latent
%   vectors; reg_1: regularization for users' activity vectors
%   init_std: latent factors are initialized as samples of zero-mean and 
%       init_std standard-deviation Gaussian distribution

[max_iter, alpha,  tol, reg_u, reg_i, reg_1, init_std] = ...
   process_options(varargin, 'max_iter', 15, 'alpha', 2, 'tol', 1e-3, 'reg_u', 0.01, ...
                   'reg_i', 0.01, 'reg_1', 0.01, 'inid_std', 0.01);

[M, N] = size(R);
U = randn(M, K) * init_std;
V = randn(N, K) * init_std;
W = R * alpha;
R01 = +(R > 0);
R01t = R01';
Wt = W';
% alternatively, we can set it as 
% W = log(1+ R/epsilon); where epsilon = 1e-8;
initgrad = GetGradient(R01, W, U, V, X, Y, reg_u, reg_i, reg_1);
tolX = tol * initgrad;
alpha = ones(M, 1);
for iter = 1:max_iter
    VtV = V' * V;
    YtV = Y' * V;
    [U, gradU ] = optimize(R01, W, U, V, VtV, YtV, X, Y, reg_u);
    UtU = U' * U;
    XtU = X' * U;
    [V, gradV ] = optimize(R01t, Wt, V, U, UtU, XtU, Y, X, reg_i);
    [X, gradX] = nnwls(R01, W, X, Y, YtY, YtV, U, V, reg_1, tolX, 30, alpha);
    gradnorm = sqrt(gradU + gradV + gradX); %norm([gradU; gradV], 'fro');
    if iter == 1
        initgrad = gradnorm;
    elseif gradnorm < tol * initgrad
            break;
    end
end
end
function [U, gradU] = optimize(R, W, Uinit, V, VtV, YtV, X, Y, reg)
[M, N] = size(R);
U = zeros(size(Uinit));
gradU = 0;
for i = 1 : M
    Wi = spdiags(W(i,:)',0, N, N);
    Vt_minus_V = V' * Wi * V + VtV + reg * eye(N, N);
    Yt_minus_V = Y' * Wi * V + YtV;
    Estimate = (W(i,:) .* R(i,:)) * V - X(i,:) * Yt_minus_V;
    U(i,:) = Vt_minus_V \ Estimate';
    gradU = gradU + sum(((Uinit(i,:) - U(i, :)) * Vt_minus_V ).^2);
end

end

function [X, gradX, alpha, iters] = nnwls(R, W, Xinit, Y, YtY, YtV, U, V, reg,...
    tol, max_iter, alpha_init)
[M, N] = size(R);
X = sparse(size(Xinit));
gradX = 0;
alpha = alpha_init;
iters = zeros(M, 1);
for u = 1:M
    x = Xinit(u, :);
    Wu = spdiags(W(u,:)', 0, N, N);
    Yt_minus_Y = Y' * Wu * Y + YtY;
    Yt_minus_V = Y' * Wu * V + YtV;
    grad_invariant = Yt_minus_V * U(u,:) - ((W(u,:) .* R(u,:)) * Y)' + reg;
    beta = 0.1;
    for iter =1:max_iter
        grad = grad_invariant + Yt_minus_Y * x;
        projnorm = norm(grad(grad<0 | x >0), 'fro');
        if projnorm < tol(u)
            break;
        end
        for step =1:20 % search step size
            xn = sparse(max(x - alpha(u)* grad', 0)); d = xn - x;
            gradd = sum(grad .* d); dQd = d' * Yt_minus_Y * d;
            suff_decr = 0.99 * gradd + 0.5 * dQd <0;
            if step==1
                decr_alpha = ~suff_decr; xp = x;
            end
            if decr_alpha
                if suff_decr
                    x = xn; break;
                else
                    alpha(u) = alpha(u) * beta;
                end
            else
                if ~suff_decr || xp == xn
                    x = xp; break;
                else
                    alpha(u) = alpha(u) / beta; xp = xn;
                end
            end
        end
    end
    iters(u) = iter;
    [I, loc, val ] = find(x);
    X = X + sparse(u*I, loc, val, M, len(x));
    gradX = gradX + norm(grad(grad<0 | x>0), 'fro') ^2;
end
end

function gradient = GetGradient(R, W, U, V, X, Y, reg_u, reg_i, reg_1)
[M, N] = size(R);
VtV = V'*V;
YtV = Y'*V;
YtY = Y'*Y;
UtU = U'*U;
XtU = X'*U;
gradient = 0;
for u = 1: M
    Wu = spdiags(W(u,:)',0, N, N);
    Vt_minus_V = V' * Wu * V + VtV + reg_u * eye(N);
    Yt_minus_V = Y' * Wu * V + YtV;
    Yt_minus_Y = Y' * Wu * Y + YtY;
    grad_U = Vt_minus_V * U(u,:)' + X(u,:) * Yt_minus_V - (W(u,:) .* R(u,:)) * V;
    grad_X = Yt_minus_V * U(u,:)' + Yt_minus_Y * X(u,:)' - ((W(u,:) .* R(u,:)) * Y)' + reg_1;
    gradient = gradient + sum(grad_U .^ 2) + sum(grad_X(grad_X<0 | X(u,:)>0) .^2);
end
for i = 1:N
    Wi = spdiags(W(:,i),0, M, M);
    Ut_minus_U = U' * Wi * U + UtU + reg_i * eye(M);
    Xt_minus_U = X' * Wi * U + XtU;
    grad_I = Ut_minus_U * V(i,:)' + Y(i,:) * Xt_minus_U  - U' * (W(:,i) .* R(:,i));
    gradient = gradient + sum(grad_I .^2);
end
gradient = sqrt(gradient);

end