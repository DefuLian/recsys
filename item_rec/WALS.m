function [ U, V, metric ] = WALS(train, K, test, varargin )
%WALS Weighted Alternative Least Sqaure for Matrix Factorization from
%implicit freedback data
%Author : Defu Lian, University of Science and Techonology of China
%   R: user-item rating matrix 
%   U, V: user and item latent factors
%   K: dimension of latent vector
%   max_iter: the maximal iteration
%   tol: tolerance for a relative stopping condition
%   reg_u, reg_i: regularization for users and items
%   init_std: latent factors are initialized as samples of zero-mean and init_std
%   standard-deviation Gaussian distribution
randn('state', 10);
R = train>0;
[M, N] = size(R);
[max_iter, reg_u, reg_i, U, V, alpha] = ...
   process_options(varargin, 'max_iter', 10, 'reg_u', 0.01, ...
                   'reg_i', 0.01, 'U', randn(M, K) * 0.01, 'V', randn(N, K)*0.01, 'alpha', 10);
W = R * alpha;
Wt = W.';
Rt = R.';
% alternatively, we can set it as 
% W = log(1+ R/epsilon); where epsilon = 1e-8;
for iter = 1:max_iter
    VtV = V.' * V + reg_u * eye(K);
    [U, gradU ] = Optimize(Rt, Wt, U, V, VtV);
    UtU = U.' * U + reg_i * eye(K);
    [V, gradV ] = Optimize(R, W, V, U, UtU);
    gradnorm = sqrt(gradU + gradV); %norm([gradU; gradV], 'fro');
    %[prec,recall] = Evaluate(R, test, U, V);
    %matk = PredictK(test, R, U, V.',500);
    %user_count = sum(test>0 & xor(R>0, test>0), 2);
    %[prec,recall] = PrecRecall(matk,user_count(user_count>0), 500);
    %fprintf('Iteration=%d, gradient norm %f, loss %f, recall@200=%f\n', iter, gradnorm, Loss(R, W, U, V), recall(200));
    fprintf('Iteration=%d, gradient norm %f, loss %f\n', iter, gradnorm, Loss(R, W, U, V));
end
[prec,recall] = Evaluate(R, test, U, V);
metric = struct('prec', prec, 'recall', recall);
end
function [U, gradU] = Optimize(R, W, U, V, VtV)
[~, M] = size(W);
gradU = 0;
Vt = V.';
for i = 1 : M
    w = W(:, i);
    r = R(:, i);
    if nnz(w) == 0
        continue;
    end
    Ind = w>0; Wi = diag(w(Ind));    %Wi = repmat(w(Ind), 1, size(V, 2));
    sub_V = V(Ind,:);
    VCV = sub_V.' * Wi * sub_V + VtV; %Vt_minus_V = sub_V.' * (Wi .* sub_V) + invariant;
    Y = Vt * (w .* r + r);
    u = VCV \ Y;
    grad = (U(i,:) - u') * VCV ;
    U(i,:) = u;
    gradU = gradU + sum(grad .^2);
end
end

%function loss = Loss(R, W, P, Q)
%[M, N] = size(R);
%K = size(P,2);
%om = ones(M, 1);
%on = ones(N, 1);
%ok = ones(K, 1);
%WR = (W .* R + R);
%R2 = R.^2;
%WR2 = om.' * ( W .* R2 + R2 ) * on;
%I = on.' * (WR.' * P .* Q) * ok;
%L = 0;
%Wt = W.';
%QtQ = Q.' * Q;
%for i=1:M
%    w = Wt(:, i);
%    p = P(i,:);
%    if nnz(w) == 0
%        continue;
%    end
%    Ind = w>0; Wi = diag(w(Ind));    %Wi = repmat(w(Ind), 1, size(V, 2));
%    sub_Q = Q(Ind,:);
%    QcQ = sub_Q.' * Wi * sub_Q + QtQ;
%    L = L + p * QcQ * p.';
%end
%loss = WR2 + L - 2 * I;
%end

