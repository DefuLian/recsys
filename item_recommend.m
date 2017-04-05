function metric = item_recommend(rec, mat, varargin)
option = 'output';
for i=1:2:length(varargin)
    if strcmp(varargin{i}, 'test') || strcmp(varargin{i}, 'split_ratio')
        option = 'heldout';
        break;
    elseif strcmp(varargin{i}, 'folds')
        option = 'crossvalid';
        break        
    end
end
if strcmp(option, 'heldout')
    metric = heldout_rec(rec, mat, @compute_score_item, varargin{:});
end
if strcmp(option, 'crossvalid')
    metric = crossvalid_rec(rec, mat, @compute_score_item, varargin{:});
end
if strcmp(option, 'output')
    metric = heldout_rec(rec, mat, @compute_score_item, 'test', sparse(size(mat,1), size(mat,2)), varargin{:});
end
end