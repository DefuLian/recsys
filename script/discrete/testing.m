dir = 'C:/Users/liand/Desktop/code/dataset';
K = 64; max_iter = 20;

dataset = 'ml10Mdata';
load(sprintf('%s/%s.mat', dir , dataset))
Traindata(Testdata>0)=0;
data = Traindata + Testdata;

algs(1).alg = @(mat,varargin) pph(mat, 'max_iter', 20, 'K', K, 'debug',false, varargin{:}); algs(1).paras = {'rating', true, 'lambda', [0.01,0.5,1,2,4,8,16]}; 
algs(2).alg = @(mat,varargin) bccf(mat, 'max_iter', 20, 'K', K, 'debug',false, varargin{:}); algs(2).paras = {'rating', true, 'lambda',  0.01*(1:2:10)};
algs(3).alg = @(mat,varargin) DCF(mat, 'max_iter', 20, 'K', K, 'debug',false, varargin{:}); algs(3).paras = {'rating', true, 'alpha', 10.^(-4:2), 'beta', 10.^(-4:2)};
algs(4).alg = @(mat,varargin) ch(mat, 'max_iter', 100, 'K', K, 'debug',false, varargin{:}); algs(4).paras = {'rating', true};
result = cell(length(algs),1);
for i=length(algs):-1:1
    [outputs{1:6}] = running(algs(i).alg, data, algs(i).paras{:});
    result{i} = outputs;
end
save(sprintf('%s/%s_result.mat',dir, dataset), 'result');

