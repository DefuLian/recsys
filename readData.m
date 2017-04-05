function [ train, test ] = readData( data_dir, zero_start )
%DataRead: read tuples from train and test files and transform them as a
%train and test sparse matrix
%   data_dir: gives the working directory of train and test files, in this
%   directory, there is a training file, named as train.txt and a testing
%   file, named as test.txt; each line in these two files consist of
%   user_id, item_id and rating; these fields are delimited by one tab
train_file = fullfile(data_dir, 'train.txt');
test_file = fullfile(data_dir, 'test.txt');
f_train = fopen(train_file);
f_test = fopen(test_file);
C_train = textscan(f_train,'%f\t%f\t%f');
C_test = textscan(f_test, '%f\t%f\t%f');
fclose(f_train);
fclose(f_test);
if zero_start
    numUsers = max(max(C_train{1}), max(C_test{1})) + 1;
    numItems = max(max(C_train{2}), max(C_test{2})) + 1;
    train = sparse(C_train{1} + 1, C_train{2} + 1, C_train{3}, numUsers, numItems);
    test = sparse(C_test{1} + 1, C_test{2} + 1, C_test{3}, numUsers, numItems);
else
    numUsers = max(max(C_train{1}), max(C_test{1}));
    numItems = max(max(C_train{2}), max(C_test{2}));
    train = sparse(C_train{1}, C_train{2}, C_train{3}, numUsers, numItems);
    test = sparse(C_test{1}, C_test{2}, C_test{3}, numUsers, numItems);
end
end

