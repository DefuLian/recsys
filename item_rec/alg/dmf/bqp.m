function [x, l] = bqp(x_init, A, b, alg)
k = size(A,1);
assert(issymmetric(A), 'matrix A should be sysmmetric');
if strcmpi(alg, 'ccd')
    [x, l] = ccd(x_init, A, b);
elseif strcmpi(alg, 'svr')
    if k > 32
        [x,l] = bqp_large(x_init, A, b, 32);
    else
        [x,l] = bqp_small(A, b);
    end
elseif strcmpi(alg, 'mix')
    [x, l] = bqp_mix(x_init, A, b, 32);
else
    error('unsupported solver');
end

end
%%% bqp_mix alternatively use ccd and bcd for optimization.
function [x,l] = bqp_mix(x_init, A, b, bsize, max_iter)
k = size(A,1);
convergent = false;
if nargin < 5
    max_iter = 500;
end
iter = 1;
x = x_init;
while ~convergent
    [x1, ~] = bqp_large(x, A, b, bsize, 1);
    [x2, l] = ccd(x1, A, b, 1);
    no_change_count = sum(x2 == x);
    x = x2;
    if iter >= max_iter || no_change_count == k
        convergent = true;
    end
    iter = iter + 1;
end
end
%%% bqp_large use block coodinate descent for optimziation
%%% parameter bsize: specifies the size of block
function [x,l] = bqp_large(x_init, A, b, bsize, max_iter)
k = size(A,1);
bnum = (k + bsize - 1) / bsize; %% number of blocks
x = x_init;
dim_index = randperm(k);
convergent = false;
if nargin < 5
    max_iter = 500;
end
iter = 1;
while(~convergent)
    no_change_count = 0;
    for biter=1:bnum
        bstart = (biter - 1) * bsize + 1;
        bend = min(biter * bsize, k);
        index = false(k,1); index(dim_index(bstart:bend)) = true;
        x_new = bqp_small(A(index, index), b(index) - A(index, ~index) * x(~index));
        no_change_count = no_change_count + sum(x(index) == x_new);
        x(index) = x_new;
    end
    if iter >= max_iter || no_change_count == k
        convergent = true;
    end
    iter = iter + 1;
end
l = dot(x, A*x) - 2* dot(b, x);
end
function [x,l] = bqp_small(A, b)
%%% binary quadratic problem: min x' A x - 2 b' x, s.t. x in {+1,-1}^k
%%% x: optimal binary vector
%%% l: the corresponding loss
[k, ~] = size(A);
C = [A, -b; -b', 0]; % cast to min x' C x, s.t. x in {+1,-1}^k+1
[~, X, ~] = psd_ip(-C); % solve min tr(CX), st. rank(X)=1 and diag(X) = e (drop rank constraint)
Xi = sign(mvnrnd(zeros(k+1,1), X, k));
loss = sum((Xi * C) .* Xi, 2);
[l, ind] = min(loss);
x = Xi(ind,:);
t = x(k+1);
x = x(1:k) * t;
end
function [x,l] = ccd(x, A, b)
x = ccd_mex(x, A, b);
l = dot(x, A*x) - 2* dot(b, x);
end