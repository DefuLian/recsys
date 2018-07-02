%dir = 'C:/Users/liand/Desktop/code/dataset';
dir = '~/data';
parpool('local',num_threads);
K = 64; 
load(sprintf('%s/%s.mat', dir , dataset))
if ~exist('data','var')
    Traindata(Testdata>0)=0;
    data = Traindata + Testdata;
end

avg_score = sum(data,2)./sum(data~=0,2); %std_score = sqrt(sum(test.^2, 2) ./ sum(test~=0,2) - avg_score); %avg_score = avg_score + 2*std_score;
avg_score = min(avg_score, max(data,[],2)-1e-3);
m = size(data,1); avg_matrix = spdiags(avg_score, 0, m, m) * (data~=0) ;
data = +(data > avg_matrix);

file_name = sprintf('%s/%s_if_results(new_ndcg).mat',dir, dataset);
if exist(file_name, 'file')
    load(file_name);
end


algs(1).alg = @(mat,varargin) pph(mat, 'max_iter', 20, 'K', K, 'debug',true, varargin{:}); algs(1).paras = {'rating', false, 'lambda', [0.01,0.5,1,2,4,8,16]}; 
algs(2).alg = @(mat,varargin) bccf(mat, 'max_iter', 20, 'K', K, 'debug',true, varargin{:}); algs(2).paras = {'rating', false, 'lambda',  0.01*(1:2:10)};
algs(3).alg = @(mat,varargin) DCF(mat, 'max_iter', 20, 'K', K, 'debug',true, varargin{:}); algs(3).paras = {'rating', false, 'alpha', 10.^(-4:2), 'beta', 10.^(-4:2)};
algs(4).alg = @(mat,varargin) ch(mat, 'max_iter', 100, 'K', K, 'debug',true, varargin{:}); algs(4).paras = {'rating', false};
if ~exist('result', 'var')
    result = cell(length(algs),1);
end
for i=length(algs):-1:1
    if ~isempty(result{i})
        continue
    end
    fprintf('algorithm: %d\n', i)
    [outputs{1:6}] = running(algs(i).alg, data, algs(i).paras{:});
    result{i} = outputs;
    save(file_name, 'result');
end

