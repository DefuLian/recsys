function [ W ] = GetWeight( mat, varargin )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[M, N] = size(mat);
[epsilon] = process_options(varargin, 'epsilon', 1e-10);
[I, J, V] = find(mat);
V_t = log10(1+ V / epsilon);
W = sparse(I, J, V_t, M, N);
end

