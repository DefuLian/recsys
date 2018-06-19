function [metric, metric_detail, best_para, paras_tuned, metrics_tuned, times_tuned] = running(method_fun, dataset, varargin)
% method_fun=(mat,varargin) 
[metric_fun, rating, para_search_mode, paras] = process_options(varargin, 'metric_fun', @(metric) metric.ndcg(1,end), 'rating', false, 'search_mode', 'grid');
if ~rating
    no_tune = all(cellfun(@length, paras(2:2:end))==1);
    if ~no_tune
        [best_para, paras_tuned, metrics_tuned, times_tuned] = hyperp_search(...
        @(varargin) item_recommend(method_fun, dataset, 'test_ratio', 0.2, varargin{:}), metric_fun, 'mode', para_search_mode, paras{:});
    else
        best_para = paras; paras_tuned = []; metrics_tuned = []; times_tuned = [];
    end
    [metric.item, metric_detail.item, time] = item_recommend(@(mat) method_fun(mat, best_para{:}), dataset, 'test_ratio', 0.2, 'times', 5);
else
    no_tune = all(cellfun(@length, paras(2:2:end))==1);
    if ~no_tune
        [best_para, paras_tuned, metrics_tuned, times_tuned] = hyperp_search(...
            @(varargin) rating_recommend(method_fun, dataset, 'test_ratio', 0.2, varargin{:}), metric_fun, 'mode', para_search_mode, paras{:});
    else
        best_para = paras; paras_tuned = []; metrics_tuned = []; times_tuned = [];
    end
    [metric.item, metric_detail.item, time] = item_recommend(@(mat) method_fun(mat, best_para{:}), dataset, 'test_ratio', 0.2, 'times', 5);
    [metric.rating, metric_detail.rating] = rating_recommend(@(mat) method_fun(mat, best_para{:}), dataset, 'test_ratio', 0.2, 'times', 5);
end
times_tuned = [times_tuned; time];
end