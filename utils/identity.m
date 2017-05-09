function [P, Q] = identity(mat, varargin)
[P, Q] = process_options(varargin, 'P', [], 'Q', []);
[M, N] = size(mat);
fprintf('identity function\n')
if ~isempty(P) 
    if size(P,2) == M && isempty(Q)
        Q = +(mat>0).';
    elseif isempty(Q)
        error('Please provide latent factors for items')
    end
elseif ~isempty(Q)
    if size(Q,2) == N
        P = +(mat>0);
    else
       error('Please provide latent factors for users')
    end
end