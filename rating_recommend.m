function metric = rating_recommend(rec, mat, varargin)
for i=1:2:length(varargin)
    if strcmp(varargin{i}, 'test') ==0 || strcmp(varargin{i}, 'split_ratio') ==0
        metric = heldout_rec(rec, mat, @compute_score_rating, varargin);
        break
    elseif strcmp(varargin{i}, 'folds') == 0
        metric = crossvalid_rec(rec, mat, @compute_score_rating, varargin);
        break
    else
        error('please input correct parameters')
    end
end
end