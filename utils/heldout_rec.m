function [metric, topkmat] = heldout_rec(rec, mat, scoring, varargin)
[test, ratio, split_mode, times, topk, cutoff, rec_opt] = process_options(varargin, 'test', [], ...
    'split_ratio', 0.8, 'split_mode', 'un', 'times', 5, 'topk', -1, 'cutoff', -1);

if cutoff<=0
    if topk>0
        cutoff = topk;
    else
        cutoff = 200;
    end
if topk > 0 && topk < cutoff
    topk = cutoff;
end
topkmat = [];
if ~isempty(test)
    % recommendation for the given dataset
    train = mat;
    %[P,Q]=iccf(train, 'K', 50, 'max_iter', 10);
    [P, Q] = rec(train, rec_opt{:});
    if nargout == 2
        [metric, topkmat] = scoring(train, test, P,  Q, topk, cutoff);
    else
        metric = scoring(train, test, P,  Q, topk, cutoff);
    end
    fns = fieldnames(metric);
    for f=1:length(fns)
        fieldname = fns{f};
        field_mean = metric.(fieldname);
        metric.(fieldname) = [field_mean; zeros(1,length(field_mean))];
    end
else
    % split mat and perform recommendation
    metric = struct();
    for t=1:times
        [train, test] = split_matrix(mat, split_mode, ratio);
        [P, Q] = rec(train, rec_opt{:});
        if strcmp(split_mode, 'i')
            ind = sum(test)>0;
            metric_time = scoring(train(:,ind), test(:,ind), P,  Q(ind,:), topk, cutoff);
        else
            metric_time = scoring(train, test, P,  Q, topk, cutoff);
        end
        fns = fieldnames(metric_time);
        for f=1:length(fns)
            fieldname = fns{f};
            if isfield(metric, fieldname)
                metric.(fieldname) = metric.(fieldname) + [metric_time.(fieldname);(metric_time.(fieldname)).^2];
            else
                metric.(fieldname) = [metric_time.(fieldname);(metric_time.(fieldname)).^2];
            end
        end
    end
    fns = fieldnames(metric);
    for f=1:length(fns)
        fieldname = fns{f};
        field = metric.(fieldname);
        field_mean = field(1,:) / times;
        field_std = sqrt(field(2,:)./times - field_mean .* field_mean);
        metric.(fieldname) = [field_mean; field_std];
    end
end

end
