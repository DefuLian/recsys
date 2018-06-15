function [metric,elapsed] = crossvalid_rec(rec, mat, scoring, varargin)
% rec: recommendation method
% mat: matrix storing records
% mode of cross validation: 
%   un: user side normal 
%   in: item side normal
%   en: entry wise normal
%   u: user
%   i: item


[folds, fold_mode, rec_opt] = process_options(varargin, 'folds', 5, 'fold_mode', 'un');

assert(folds>0)



mat_fold = kFolds(mat, folds, fold_mode);
metric = struct();
elapsed = zeros(1,2);
for i=1:folds
    test = mat_fold{i};
    train = mat - test;
    tic;[P, Q] = rec(train, rec_opt{:}); elapsed(1) = elapsed(1) + toc/folds;
    tic;
    if strcmp(fold_mode, 'i') % in this mode, only those items within the same fold are required for comparison
        ind = sum(test)>0;
        metric_fold = scoring(train(:,ind), test(:,ind), P,  Q(ind,:));
    else
        metric_fold = scoring(train, test, P,  Q, topk, cutoff);
    end
    elapsed(2) = elapsed(2) + toc/folds;
    fns = fieldnames(metric_fold);
    for f=1:length(fns)
        fieldname = fns{f};
        if isfield(metric, fieldname)
            metric.(fieldname) = metric.(fieldname) + [metric_fold.(fieldname);(metric_fold.(fieldname)).^2];
        else
            metric.(fieldname) = [metric_fold.(fieldname);(metric_fold.(fieldname)).^2];
        end
    end
end
fns = fieldnames(metric);
for f=1:length(fns)
    fieldname = fns{f};
    field = metric.(fieldname);
    field_mean = field(1,:) / folds;
    field_std = sqrt(field(2,:)./folds - field_mean .* field_mean);
    metric.(fieldname) = [field_mean; field_std];
end
end

