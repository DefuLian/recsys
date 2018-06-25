function [eval_summary,eval_detail,elapsed] = rating_recommend(rec, mat, varargin)
[cutoff_rating, cutoff, topk, varargin] = process_options(varargin, 'cutoff_rating', 20, 'cutoff', -1, 'topk', -1);
if topk > 0 && cutoff > 0
    topk = cutoff;
elseif cutoff<=0
    if topk>0
        cutoff = topk;
    else
        cutoff = 200;
    end
end
option = 'output';
for i=1:2:length(varargin)
    if strcmp(varargin{i}, 'test') || strcmp(varargin{i}, 'test_ratio')
        option = 'heldout';
        break;
    elseif strcmp(varargin{i}, 'folds')
        option = 'crossvalid';
        break;        
    end
end
if strcmp(option, 'heldout')
    [eval_summary,eval_detail,elapsed] = heldout_rec(rec, mat, @evaluate, varargin{:});
end
if strcmp(option, 'crossvalid')
    [eval_summary,eval_detail,elapsed] = crossvalid_rec(rec, mat, @evaluate, varargin{:});
end
if strcmp(option, 'output')
    [eval_summary,eval_detail,elapsed] = heldout_rec(rec, mat, @evaluate, 'test', sparse(size(mat,1), size(mat,2)), varargin{:});
end
function metric = evaluate(tr,te,p,q)
    metric1 = evaluate_rating(te,p,q,cutoff_rating);
    te(te<0) = 0; %% negative rating scores are ignored in testing.
    metric2 = evaluate_item(tr,te,p,q,topk,cutoff);
    names = [fieldnames(metric1); fieldnames(metric2)];
    metric = cell2struct([struct2cell(metric1); struct2cell(metric2)], names, 1);
end
end
