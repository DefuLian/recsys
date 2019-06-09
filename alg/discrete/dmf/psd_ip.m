function [phi, X, y, converge] = psd_ip(L, varargin)
% solves: max trace(LX) s.t. X psd, diag(X) = b; b = ones(n,1)/4
% min b'y s.t. Diag(y) - L psd, y unconstrained,
% input: L ... symmetric matrix
% output: phi ... optimal value of primal, phi =trace(LX)
% X ... optimal primal matrix
% y ... optimal dual vector
% call: [phi, X, y] = psd_ip( L);
[n, ~] = size(L); % problem size
[verbose, b, digits, max_iter] = process_options(varargin, 'verbose', false,...
    'b', ones(n, 1), 'precision', 6, 'max_iter', 200);
%digits = 6; % 6 significant digits of phi
%b = ones(n,1 )/4; % any b>0 works just as well
X = diag(b); % initial primal matrix is pos. def.
y = sum(abs(L))' * 1.1; % initial y is chosen so that
Z = diag(y) - L; % initial dual slack Z is pos. def.
phi = b'* y; % initial dual
psi = L(:)' * X(:); % and primal costs
mu = Z(:)' * X(:) / (2*n); % initial complementarity
if verbose
    disp('iter alphap alphad gap lower upper');
end
prev_gap = inf;
min_alpha = 1e-20;
converge = true;
for iter = 1:max_iter
    cur_gap = phi - psi;
    %if cur_gap < 1.49*10^(-digits) || abs(cur_gap - prev_gap) < prev_gap * 1e-3
    if cur_gap < max([1,abs(phi)]) * 10^(-digits) || abs(cur_gap - prev_gap) < prev_gap * 1e-4
        break
    end
    prev_gap = cur_gap;
    Zi = inv(Z); % inv(Z) is needed explicitly, Z is psd
    Zi = (Zi + Zi')/2;
    dy = (Zi .* X) \ (mu * diag(Zi) - b); % solve for dy, the Hadamard product of two positive definite matrices is also a positive definite matrix
    dX = -Zi * diag(dy) * X + mu * Zi - X; % back substitute for dX
    dX = (dX + dX')/2; % symmetrize
    % line search on primal
    alphap = 1; % initial steplength
    [~, posdef] = chol( X + alphap * dX ); % test if pos.def
    while posdef > 0
        alphap = alphap * .8;
        if alphap < min_alpha
            break
        end
        [~, posdef] = chol( X + alphap * dX );
    end
    if posdef > 0
        converge = false;
        break;
    end
    if alphap < 1
        alphap = alphap * .95; 
    end % stay away from boundary
    
    % line search on dual; dZ is handled implicitly: dZ = diag(dy);
    alphad = 1;
    [~, posdef] = chol(Z + alphad * diag(dy));
    while posdef > 0
        alphad = alphad * .8;
        if alphad < min_alpha
            break;
        end
        [~, posdef] = chol(Z + alphad * diag(dy));
    end
    if posdef > 0
        converge = false;
        break
    end
    if alphad < 1
        alphad = alphad * .95; 
    end
    % update
    X = X + alphap * dX;
    y = y + alphad * dy;
    Z = Z + alphad * diag(dy);
    mu = X(:)' * Z(:) / (2*n);
    if alphap + alphad > 1.8
        mu = mu/2; 
    end % speed up for long steps
    phi = b' * y; 
    psi = L(:)' * X(:);
    % display current iteration
    %disp([ iter alphap alphad (phi-psi) psi phi ]);
    if verbose
        fprintf('%2d %.2f %.2f %10.3f %.3f %.3f\n', iter, alphap, alphad, (phi-psi), psi, phi);
    end
end % end of main loop
end
