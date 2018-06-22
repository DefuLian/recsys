function [eval_summary, eval_detail,elapsed] = crossvalid_rec(rec, mat, scoring, varargin)
% rec: recommendation method
% mat: matrix storing records
% mode of cross validation: 
%   un: user side normal 
%   in: item side normal
%   en: entry wise normal
%   u: user
%   i: item


[folds, fold_mode, seed, rec_opt] = process_options(varargin, 'folds', 5, 'fold_mode', 'un', 'seed', 1);

assert(folds>0)


rng(seed);
mat_fold = kFolds(mat, folds, fold_mode);
eval_detail = struct();
elapsed = zeros(folds,2);
metric_folds = cell(folds,1);
parfor i=1:folds
    test = mat_fold{i};
    train = mat - test;
    tic;[P, Q] = rec(train, rec_opt{:}); t1 =  toc;
    tic;
    if strcmp(fold_mode, 'i') % in this mode, only those items within the same fold are required for comparison
        ind = sum(test)>0;
        metric_folds{i} = scoring(train(:,ind), test(:,ind), P,  Q(ind,:));
    else
        metric_folds{i} = scoring(train, test, P,  Q);
    end
    t2 = toc;
    elapsed(i,:) = [t1,t2];
    
end
for i=1:folds
    metric_fold = metric_folds{i};
    fns = fieldnames(metric_fold);
    for f=1:length(fns)
        fieldname = fns{f};
        if isfield(eval_detail, fieldname)
            %metric.(fieldname) = metric.(fieldname) + [metric_fold.(fieldname);(metric_fold.(fieldname)).^2];
            eval_detail.(fieldname) = [eval_detail.(fieldname); metric_fold.(fieldname)];
        else
            %metric.(fieldname) = [metric_fold.(fieldname);(metric_fold.(fieldname)).^2];
            eval_detail.(fieldname) = metric_fold.(fieldname);
        end
    end
end
fns = fieldnames(eval_detail);
for f=1:length(fns)
    fieldname = fns{f};
    field = eval_detail.(fieldname);
    %field_mean = field(1,:) / folds;
    %field_std = sqrt(field(2,:)./folds - field_mean .* field_mean);
    eval_summary.(fieldname) = [mean(field,1); std(field,0,1)];
end
elapsed = mean(elapsed,1);
end

