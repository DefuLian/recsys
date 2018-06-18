function [opt_para, para_all, result, times] = hyperp_search(alg_func, metric_func, varargin)

names = varargin(1:2:length(varargin));
ranges = varargin(2:2:length(varargin));
total_ele = prod(cellfun(@(c) length(c), ranges));
[Ind{1:length(ranges)}] = ndgrid(ranges{:});
Indmat = cell2mat(cellfun(@(mat) mat(:), Ind, 'UniformOutput', false));
max_metric = 0;

para_all = zeros(total_ele, length(names));
metrics = cell(total_ele, 1);
times = zeros(total_ele, 2);
for iter_ele=1:total_ele
    val = Indmat(iter_ele,:);
    para = cell(length(names)*2, 1);
    for n=1:length(names)
        para((2*n-1):(2*n)) = {names{n}, val(n)};
    end
    [metric, ~, times(iter_ele,:)] = alg_func(para{:});
    cur_metric = metric_func(metric);
    para_all(iter_ele,:) = val;
    metrics{iter_ele} = metric;
    if max_metric < cur_metric
        opt_para = para;
        max_metric = cur_metric;
    end
end

metrics_array = [metrics{:}];
result = struct();
fns = fieldnames(metrics{1});
for f=1:length(fns)
    mm = cell2mat({metrics_array.(fns{f})}');
    result.(fns{f}) = mm(1:2:end,:);
end
end