function [x,l] = bqp(A, b)
%%% binary quadratic problem: min x' A x - 2 b' x, s.t. x in {+1,-1}^k
%%% x: optimal binary vector
%%% l: the corresponding loss
[k, ~] = size(A);
assert(issymmetric(A), 'matrix A should be sysmmetric');
C = [A, -b; -b', 0]; % cast to min x' C x, s.t. x in {+1,-1}^k+1
[~, X, ~] = psd_ip(-C); % solve min tr(CX), st. rank(X)=1 and diag(X) = e (drop rank constraint)
Xi = sign(mvnrnd(zeros(k+1,1), X, k));
loss = sum((Xi * C) .* Xi, 2);
[l, ind] = min(loss);
x = Xi(ind,:);
t = x(k+1);
x = x(1:k) * t;
end