function evalout = item_recommend(rec, mat, varargin)
option = 'output';

for i=1:2:length(varargin)
    if strcmp(varargin{i}, 'test') || strcmp(varargin{i}, 'split_ratio')
        option = 'heldout';
        break;
    elseif strcmp(varargin{i}, 'folds')
        option = 'crossvalid';
        break;        
    end
end
if strcmp(option, 'heldout')
    evalout = heldout_rec(rec, mat, @evaluate_item, varargin{:});
end
if strcmp(option, 'crossvalid')
    evalout = crossvalid_rec(rec, mat, @evaluate_item, varargin{:});
end
if strcmp(option, 'output')
    evalout = heldout_rec(rec, mat, @evaluate_item, 'test', sparse(size(mat,1), size(mat,2)), varargin{:});
end
end