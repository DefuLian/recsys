function [evalout, elapsed] = rating_recommend(rec, mat, varargin)
[cutoff, varargin] = process_options(varargin, 'cutoff', 10);
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
    [evalout,elapsed] = heldout_rec(rec, mat, @(tr,te,p,q) evaluate_rating(te,p,q,cutoff), varargin{:});
end
if strcmp(option, 'crossvalid')
    [evalout,elapsed] = crossvalid_rec(rec, mat, @(tr,te,p,q) evaluate_item(te,p,q,cutoff), varargin{:});
end
if strcmp(option, 'output')
    [evalout,elapsed] = heldout_rec(rec, mat, @(tr,te,p,q) evaluate_item(te,p,q,cutoff), 'test', sparse(size(mat,1), size(mat,2)), varargin{:});
end
end