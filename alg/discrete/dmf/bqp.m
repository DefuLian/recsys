function [x, l] = bqp(x_init, A, b, alg, max_iter, bsize)
%function [x, l] = bqp(x_init, A, b, varargin)
%[max_iter, bsize, alg] = process_options(varargin, 'max_iter', 500,'blocksize', 32, 'alg', 'bcd');
k = size(A,1);
%assert(issymmetric(A), 'matrix A should be sysmmetric');
if strcmpi(alg, 'ccd')
    [x, l] = ccd(x_init, A, b, max_iter);
elseif strcmpi(alg, 'svr')
    [x,l] = bqp_small(A, b);
elseif strcmpi(alg, 'bcd')
    if k > 32
        [x,l] = bqp_large(x_init, A, b, bsize, max_iter);
    else
        error('the dimension is small, please use "svr"');
    end
elseif strcmpi(alg, 'mix')
    [x, l] = bqp_mix(x_init, A, b, bsize, max_iter);
else
    error('unsupported solver');
end

end

function [x,l] = bqp_tiny(A, b)
[k, ~] = size(A);
f = @(v,f) dec2bin(v, f);
m = num2cell(f(0:(2^k-1), k));
m1 = cellfun(@(x) str2double(x), m);
m = cell2mat({m1});
m = m * 2 - 1;
loss = sum((m * A) .* m, 2) - 2 * m * b;
[l, ind] = min(loss);
x = m(ind,:);
end

%%% bqp_mix alternatively use ccd and bcd for optimization.
function [x,l] = bqp_mix(x, A, b, bsize, max_iter)
k = size(A,1);
convergent = false;
iter = 1;
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
function [x,l] = bqp_large(x, A, b, bsize, max_iter)
k = size(A,1);
bnum = (k + bsize - 1) / bsize; %% number of blocks
dim_index = randperm(k);
convergent = false;
iter = 1;
prev_loss = inf;
while(~convergent)
    %no_change_count = 0;
    
    %dim_index = 1:k;
    for biter=1:bnum
        bstart = (biter - 1) * bsize + 1;
        bend = min(biter * bsize, k);
        index = false(k,1); index(dim_index(bstart:bend)) = true;
        if bsize > 3
            [x_new,~,converge] = bqp_small(A(index, index), b(index) - A(index, ~index) * x(~index));
            if ~converge
                [x_new_1,l_1] = ccd(x(index), A(index, index), b(index) - A(index, ~index) * x(~index), 5);
                [x_new_2,l_2] = ccd(x_new, A(index, index), b(index) - A(index, ~index) * x(~index), 5);
                if l_1 < l_2
                    x_new = x_new_1;
                else
                    x_new = x_new_2;
                end
            end
        else
            x_new = bqp_tiny(A(index, index), b(index) - A(index, ~index) * x(~index));
        end
        %no_change_count = no_change_count + sum(x(index) == x_new.');
        x(index) = x_new;
    end
    current_loss = sum(x .* (A*x)) - 2* sum(b.* x);
    
    if iter >= max_iter || abs(current_loss - prev_loss) / current_loss < 1e-6
        convergent = true;
    end
    prev_loss = current_loss;
    iter = iter + 1;
end
%l = dot(x, A*x) - 2* dot(b, x);
l = sum(x .* (A*x)) - 2* sum(b.* x);
end
function [x, l, converge] = bqp_small(A, b)
%%% binary quadratic problem: min x' A x - 2 b' x, s.t. x in {+1,-1}^k
%%% x: optimal binary vector
%%% l: the corresponding loss
[k, ~] = size(A);
C = [A, -b; -b', 0]; % cast to min x' C x, s.t. x in {+1,-1}^k+1
[~, X, ~, converge] = psd_ip(-C); % solve min tr(CX), st. rank(X)=1 and diag(X) = e (drop rank constraint)
%[~, X, converge] = sdp_bqp(C);

Xi = sign(mvnrnd(zeros(k+1,1), X, k));
loss = sum((Xi * C) .* Xi, 2);
[l, ind] = min(loss);
x = Xi(ind,:);
t = x(k+1);
x = x(1:k) * t;

end
function [x,l] = ccd(x, A, b, max_iter)
x = ccd_bqp_mex(x, A, b, max_iter);
%l = dot(x, A*x) - 2* dot(b, x);
l = sum(x .* (A*x)) - 2* sum(b.* x);
end