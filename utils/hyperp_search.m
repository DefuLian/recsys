function [opt_para, para_all, result, times] = hyperp_search(alg_func, metric_func, varargin)
[mode, opt] = process_options(varargin, 'mode', 'grid');
if strcmpi(mode, 'grid')
    [opt_para, para_all, result, times] = grid_search(alg_func, metric_func, opt{:});
elseif strcmpi(mode, 'seq')
    [opt_para, para_all, result, times] = seq_search(alg_func, metric_func, opt{:});
else
    error('unsupported mode')
end
end
function [opt_para, para_all, result, times] = seq_search(alg_func, metric_func, varargin)
num = length(varargin)/2;
opt_para = [varargin(1:2:end);num2cell(zeros(1,num))];
opt_para = struct(opt_para{:});
total_ele = sum(cellfun(@length, varargin(2:2:end)));
para_all = zeros(total_ele, num);
metrics = cell(total_ele, 1);
times = zeros(total_ele, 2);
max_metric = 0;
iter_ele = 1;
for i=1:num
     name = varargin{2*i-1}; values = varargin{2*i};
     best_para = 0;
     for v = values
         opt_para.(name) = v;
         para_all(iter_ele,:) = struct2array(opt_para);
         para = [fieldnames(opt_para),struct2cell(opt_para)]';
         [metric,~,times(iter_ele,:)] = alg_func(para{:});
         cur_metric = metric_func(metric);
         metrics{iter_ele} = metric;
         if max_metric < cur_metric
            best_para = v;
            max_metric = cur_metric;
         end
         iter_ele = iter_ele + 1;
     end
     opt_para.(name) = best_para;
end
opt_para = [fieldnames(opt_para),struct2cell(opt_para)]';
opt_para = opt_para(:);
metrics_array = [metrics{:}];
result = struct();
fns = fieldnames(metrics{1});
for f=1:length(fns)
    mm = cell2mat({metrics_array.(fns{f})}');
    result.(fns{f}) = mm(1:2:end,:);
end

end
function [opt_para, para_all, result, times] = grid_search(alg_func, metric_func, varargin)
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