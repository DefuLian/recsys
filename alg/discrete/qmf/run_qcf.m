function run_qcf(dataset_path, result_dir, isimplicit)

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
    file_name = sprintf('%s/%s_result_qcf.mat',result_dir, dataset_name);
else
    file_name = sprintf('%s/%s_result_if_qcf.mat',result_dir, dataset_name);
end
if exist(file_name, 'file')
    load(file_name);
end

K = 64;
algs(1).alg = @(mat,varargin) iccf(mat, 'max_iter', 20, 'K', K, varargin{:}); algs(1).paras = {'alpha', [5, 10, 25, 50, 100, 250, 500]}; 
algs(2).alg = @(mat,varargin) qcf_init_pq(mat, 'max_iter', 20, 'K', K, varargin{:}); 
algs(3).alg = @(mat,varargin) qcf_init_opq(mat, 'max_iter', 20, 'K', K, varargin{:}); 
algs(4).alg = @(mat,varargin) qcf(mat, 'max_iter', 20, 'K', K, varargin{:}); 

if ~exist('result', 'var')
    result = cell(length(algs),1);
end

for i=1:length(algs)
    if ~isempty(result{i})
        continue
    end
    fprintf('algorithm: %d\n', i)
    if i==1
        [outputs{1:6}] = running(algs(i).alg, data, algs(i).paras{:});
    else
        [outputs{1:6}] = running(algs(i).alg, data, result{1}{3}{:});
    end
    result{i} = outputs;
    save(file_name, 'result');
end

end

