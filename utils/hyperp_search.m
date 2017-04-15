function [max_val, allmetric] = hyperp_search(alg_func, metric_func, varargin)

names = varargin(1:2:length(varargin));
ranges = varargin(2:2:length(varargin));
total_ele = prod(cellfun(@(c) length(c), ranges));
[Ind{1:length(ranges)}] = ndgrid(ranges{:});
Indmat = cell2mat(cellfun(@(mat) mat(:), Ind, 'UniformOutput', false));
max_metric = 0;
allmetric = zeros(total_ele, 1+length(names));
for iter_ele=1:total_ele
    val = Indmat(iter_ele,:);
    para = cell(length(names)*2, 1);
    for n=1:length(names)
        para((2*n-1):(2*n)) = {names{n}, val(n)};
    end
    metric = alg_func(para{:});
    cur_metric = metric_func(metric);
    allmetric(iter_ele,:) = [val, cur_metric];
    if max_metric < cur_metric
        max_val = para;
        max_metric = cur_metric;
    end
end
end