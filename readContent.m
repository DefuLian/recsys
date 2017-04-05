function [ mat ] = readContent( fileName, varargin )
%Read item content from fileName
%   mat: returning sparse matrix of item content
[nrows, ncols, sep, zero_start] = ...
   process_options(varargin, 'nrows', -1, 'ncols', -1, 'sep', '\t', 'zero_start', true);       

data = dlmread(fileName, sep);

if zero_start
    nrows = max(max(data(:,1)) + 1, nrows);
    ncols = max(max(data(:,2)) + 1, ncols);
    mat = sparse(data(:,1) + 1, data(:,2) + 1, data(:,3), nrows, ncols);
else
    nrows = max(max(data(:,1)), nrows);
    ncols = max(max(data(:,2)), ncols);
    mat = sparse(data(:,1), data(:,2), data(:,3), nrows, ncols);
end
end



