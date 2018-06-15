mex -largeArrayDims dcmf_all_mex.cpp
mex -largeArrayDims dcmf_init_all_mex.cpp
%mex -largeArrayDims -g DCDmex.c
DCMFinit_fast(train, zeros(size(train,1),0), feature, 8, 'maxItr',5,'eta',[0.01 500000], 'debug', 1, 'test', test);

dataset = 'movie';
load(sprintf('E:/data/data/data_split_each_user55/%s/train_%s.mat',dataset,dataset));
load(sprintf('E:/data/data/data_split_each_user55/%s/test_%s.mat',dataset,dataset));
mv = (max(max(train))+min(min(train(train>0))))/2;
train = NegativeSample(train>mv, 1);
test = test>mv;
save(sprintf('E:/data/data/classification/%s/data_%s.mat',dataset,dataset),'train','test');
[I,J,V] = find(train);
dlmwrite(sprintf('E:/data/data/classification/%s/train_%s.txt',dataset,dataset),[I-1,J-1,V],'\t');
[I,J,V] = find(test);
dlmwrite(sprintf('E:/data/data/classification/%s/test_%s.txt',dataset,dataset),[I-1,J-1,V],'\t');