function [metric, metric_detail, best_para, paras_tuned, metrics_tuned, times_tuned] = running(method_fun, dataset, varargin)
[rating_tune, para_search_mode, paras] = process_options(varargin, 'rating', false, 'search_mode', 'grid');
maxS = max(max(dataset));
minS = min(dataset(dataset~=0));
if maxS - minS > 1e-3
    rating = true;
else
    rating = false;
end
if rating_tune && rating
    metric_fun = @(metric) metric.rating_ndcg(1,1);
else
    if rating
        metric_fun = @(metric) metric.item_ndcg_score(1,end);
    else
        metric_fun = @(metric) metric.item_ndcg(1,end);
    end
end
no_tune = all(cellfun(@length, paras(2:2:end))==1);
if ~no_tune
    if rating
        [best_para, paras_tuned, metrics_tuned, times_tuned] = hyperp_search(...
            @(varargin) rating_recommend(method_fun, dataset, 'test_ratio', 0.2, varargin{:}), metric_fun, 'mode', para_search_mode, paras{:});
    else
        [best_para, paras_tuned, metrics_tuned, times_tuned] = hyperp_search(...
        @(varargin) item_recommend(method_fun, dataset, 'test_ratio', 0.2, varargin{:}), metric_fun, 'mode', para_search_mode, paras{:});
    end
else
    best_para = paras; paras_tuned = []; metrics_tuned = []; times_tuned = [];
end
if rating
    [metric, metric_detail, time] = rating_recommend(@(mat) method_fun(mat, best_para{:}), dataset, 'test_ratio', 0.2, 'times', 5);
else
    [metric, metric_detail, time] = item_recommend(@(mat) method_fun(mat, best_para{:}), dataset, 'test_ratio', 0.2, 'times', 5);
end
times_tuned = [times_tuned; time];
end