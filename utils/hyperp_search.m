function [opt_para, para_all, result, times] = hyperp_search(alg_func, metric_func, varargin)
[mode, opt] = process_options(varargin, 'mode', 'grid');
if strcmpi(mode, 'grid')
    [opt_para, para_all, result, times] = parallel_grid_search(alg_func, metric_func, opt{:});
elseif strcmpi(mode, 'seq')
    [opt_para, para_all, result, times] = parallel_seq_search(alg_func, metric_func, opt{:});
else
    error('unsupported mode')
end
end
function [opt_para, para_all, result, times] = parallel_grid_search(alg_func, metric_func, varargin)
names = varargin(1:2:length(varargin));
ranges = varargin(2:2:length(varargin));
total_ele = prod(cellfun(@(c) length(c), ranges));
[Ind{1:length(ranges)}] = ndgrid(ranges{:});
Indmat = cell2mat(cellfun(@(mat) mat(:), Ind, 'UniformOutput', false));
nn = cell(total_ele,1); [nn{:}]=deal(names);
paras = cellfun(@(x,y) [y;num2cell(x)], num2cell(Indmat,2), nn, 'UniformOutput', false);
para_all = cell(total_ele, length(names)*2);
for i=1:total_ele
    para_all(i,:) = paras{i}(:);
end
metrics = cell(total_ele, 1);
times = zeros(total_ele, 2);
parfor it =1:total_ele
    [metrics{it}, ~, times(it,:)] = alg_func(para_all{it,:});
end
[~, idx] = max(cellfun(@(x) metric_func(x), metrics));
opt_para = para_all(idx,:);

metrics_array = [metrics{:}];
result = struct();
fns = fieldnames(metrics{1});
for f=1:length(fns)
    mm = cell2mat({metrics_array.(fns{f})}');
    result.(fns{f}) = mm(1:2:end,:);
end

end

function [opt_para, para_all, result, times] = parallel_seq_search(alg_func, metric_func, varargin)
num = length(varargin)/2;
opt_para = [varargin(1:2:end);num2cell(zeros(1,num))];
opt_para = struct(opt_para{:});
total_ele = sum(cellfun(@length, varargin(2:2:end)));
para_all = cell(total_ele, num*2);
metrics = cell(total_ele, 1);
times = zeros(total_ele, 2);
max_metric = 0;
start = 0;
for i=1:num
     name = varargin{2*i-1}; values = varargin{2*i};
     default_value = opt_para.(name);
     for j = 1:length(values)
         opt_para.(name) = values(j);
         para = [fieldnames(opt_para),struct2cell(opt_para)]';
         para_all(j+start,:) = para(:);
     end
     opt_para.(name) = default_value;
     parfor j = 1:length(values)
         [metrics{j+start},~,times(j+start,:)] = alg_func(para_all{j+start,:});
     end
     [max_m,idx] = max(cellfun(@(m) metric_func(m), metrics((start+1):(start+length(values)))));
     if max_m > max_metric
         opt_para.(name) = values(idx);
         max_metric = max_m;
     end
     start = start + length(values);
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
