function [ U, V ] = ALS(R, K, varargin)
% Regularized Matrix Factorization using Alternative Least Sqaure
% Author : Defu Lian, University of Science and Techonology of China
%   R: M x N rating matrix
%   U, V: user and item latent vector
%   K: dimension of latent factors
%   max_iter: the maximal iteration
%   tol: tolerance for a relative stopping condition
%   reg_u, reg_i: regularization for users and items
%   init_std: latent factors are initialized as samples of zero-mean and init_std
%       standard-deviation Gaussian distribution
[max_iter, tol, reg_u, reg_i, init_std] = ...
   process_options(varargin, 'max_iter', 15, 'tol', 1e-3, 'reg_u', 0.01, ...
                   'reg_i', 0.01, 'inid_std', 0.01);
[M, N] = size(R);
U = randn(M, K) * init_std;
V = randn(N, K) * init_std;
VtV = V' * V + reg_u * eye(K);
RV = R * V;
gradU = U * VtV  - RV;
gradV = V * (U' * U + reg_i ) - R' * U;
initgrad = norm([gradU; gradV], 'fro');
fprintf('Init gradient norm %f\n', initgrad);
for iter = 1:max_iter
    U = RV / VtV;
    UtU = U' * U + reg_i * eye(K);
    RtU = R' * U;
    V = RtU / UtU;
    gradV = V * UtU - RtU;
    VtV = V' * V + reg_u * eye(K);
    RV = R * V;
    gradU = U * VtV - RV;
    grad = norm([gradU; gradV], 'fro');
    fprintf('Iteration=%d, gradient norm %f\n', iter, grad);
    if grad < tol * initgrad
        break;
    end
    
end
end

