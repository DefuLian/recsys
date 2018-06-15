function [evalout, elapsed] = item_recommend(rec, mat, varargin)
[topk, cutoff, varargin] = process_options(varargin, 'topk', -1, 'cutoff', -1);
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
    [evalout,elapsed] = heldout_rec(rec, mat, @(tr,te,p,q) evaluate_item(tr,te,p,q,topk,cutoff), varargin{:});
end
if strcmp(option, 'crossvalid')
    [evalout,elapsed] = crossvalid_rec(rec, mat, @(tr,te,p,q) evaluate_item(tr,te,p,q,topk,cutoff), varargin{:});
end
if strcmp(option, 'output')
    [evalout,elapsed] = heldout_rec(rec, mat, @(tr,te,p,q) evaluate_item(tr,te,p,q,topk,cutoff), 'test', sparse(size(mat,1), size(mat,2)), varargin{:});
end
end