function run_dmf(dataset_path, result_dir, isimplicit)
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
    file_name = sprintf('%s/%s_result_dmf.mat',result_dir, dataset_name);
else
    file_name = sprintf('%s/%s_result_if_dmf.mat',result_dir, dataset_name);
end
if exist(file_name, 'file')
    load(file_name);
end

K = 64;
algs(1).alg = @(mat,varargin) pph(mat, 'max_iter', 20, 'K', K, 'debug',true, varargin{:}); algs(1).paras = {'rating', ~isimplicit, 'lambda', [0.01,0.5,1,2,4,8,16]}; 
algs(2).alg = @(mat,varargin) bccf(mat, 'max_iter', 20, 'K', K, 'debug',true, varargin{:}); algs(2).paras = {'rating', ~isimplicit, 'lambda',  0.01*(1:2:10)};
algs(3).alg = @(mat,varargin) DCF(mat, 'max_iter', 20, 'K', K, 'debug',true, varargin{:}); algs(3).paras = {'rating', ~isimplicit, 'alpha', 10.^(-4:2), 'beta', 10.^(-4:2)};
algs(4).alg = @(mat,varargin) ch(mat, 'max_iter', 100, 'K', K, 'debug',true, varargin{:}); algs(4).paras = {'rating', ~isimplicit};
algs(5).alg = @(mat,varargin) dmf(mat, 'K', 64, 'max_iter',20, 'islogit', true, varargin{:}); algs(5).paras = {'rho',10.^(-6:0), 'alpha', 10.^(-4:2), 'beta',10.^(-4:2)};
algs(6).alg = @(mat,varargin) dmf(mat, 'K', 64, 'max_iter',20, varargin{:}); algs(6).paras = {'rho', 10.^(-6:0), 'alpha', 10.^(-4:2), 'beta',10.^(-4:2)};



if ~exist('result', 'var')
    result = cell(length(algs),1);
end

for i=1:length(algs)
    if ~isempty(result{i})
        continue
    end
    fprintf('algorithm: %d\n', i)
    if i>=5
        [outputs{1:6}] = running(algs(i).alg, data, 'search_mode', 'seq', algs(i).paras{:});
    else
        [outputs{1:6}] = running(algs(i).alg, data, algs(i).paras{:});
    end
    result{i} = outputs;
    save(file_name, 'result');
end

end

