function qcf_dim_tune(dataset_path, result_dir, isimplicit)
[~,dataset_name,~] = fileparts(dataset_path);
load(dataset_path);
if ~isexplict(data)
    isimplicit = true;
else
    if isimplicit
        data = explicit2implicit(data);
    end
end
if ~isimplicit
    file_name = sprintf('%s/%s_qcf_dim_tune.mat',result_dir, dataset_name);
else
    file_name = sprintf('%s/%s_qcf_if_dim_tune.mat',result_dir, dataset_name);
end
if exist(file_name, 'file')
    return
end

alg = @(mat,varargin) qcf(mat, 'max_iter', 10, varargin{:});
para = {'K', [16,32,64,128,256],'alpha', [5, 10, 25, 50, 100]};
clear('result')
if ~isimplicit
    [result{1:4}] = hyperp_search(...
            @(varargin) rating_recommend(alg, data, 'test_ratio', 0.2, varargin{:}), @(metric) metric.item_ndcg_score(1,end), para{:});
else
    [result{1:4}] = hyperp_search(...
            @(varargin) item_recommend(alg, data, 'test_ratio', 0.2, varargin{:}), @(metric) metric.item_ndcg(1,end), para{:});
end
save(file_name, 'result');
end

