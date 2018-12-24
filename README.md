# MRec: Matlab-based Recommender System

MRec is an academic project. It targets for item recommendation from implicit feedback and rating prediction based recommendation. It supports cross validation and holdout evaluation. It includes several important algorithms tailored for implicit feedback, such as WRMF[1], content-aware collaborative filtering for implicit feedback[3], GWMF (graph regularized weighted matrix factorization). Also it includes some state-of-the-art algorithm for location recommendation, such as IRenMF[4], GeoMF[2]. In the future, we will further implement algorithms including Bayesian personalized ranking based matrix factorization, non-negative matrix factorization, Poisson Matrix Factorization.

## Data Preprocessing
The dataset are assumed to be saved in a format file, including columns of users, items and scores (count|rating). 
```txt
u1	i1	5
u2	i2	4
```

If dataset has been split into train.txt and test.txt, you can load data via
```matlab
[train, test] = readData('path/to/file', 1); # 1 indicating indexes of users and items start with zero
```
If dataset is not split, we can load data via
```matlab
data = readContent('path/to/file'); # zero_start is default
```
after that, you can manually split data into train and test
```matlab
[train, test] = split_matrix(data, 'un', 0.8); # mode='un' and ratio=0.8
```
Ratio corresponds to the ratio of training set. There are five types of matrix split methods, by specified by 'mode'.

	* 'un' (default) user-oriented split, splitting each user data into 'folds' folds.
	* 'in' item-oriented split, 
	* 'en' entry-oriented split. 
	* 'u' splits users into 'folds' folds, which is used for cold-start user evaluation
	* 'i' splits items into 'folds' folds, which is used for cold-start item evaluation.
	
## Running
The main portal is **item_recommend** for implicit feedback and **rating_recommend** for explicit feedback. It is very simple to use. The following two parameters are mandatory to these two portals

* a handle of algorithm, which take a user-item matrix as mandatory input, and other options by name-value pairs. 
* a user-item preference matrix.

See [test/dataset/test_script.m](https://github.com/DefuLian/recsys/blob/master/test/dataset/test_script.m) for some examples of how to use this portal

Three types of running schemes are available, including cross validation and holdout evaluation. 


 1. The following command specifies cross validation, where *folds* means the number of folds. And it returns *summary* which include recall, precision, ndcg, map, auc, mpr (mean percentile rank), and 'elapsed'. 'elapsed' records training and testing time. There are two rows in *summary*, corresponding to mean and std of cross validation. For example, summary.item_recall(1,20) is the averaged recall@20 while summary.item_recall(2,20) is the standard deviation of cross validation.
```matlab
[summary, detail, elapsed] = item_recommend(@(mat) iccf(mat, 'K', 50, 'max_iter', 20), data, 'folds', 5); 
```
  which can be simplified as
```matlab
[summary, detail, elapsed] = item_recommend(@iccf, data, 'folds', 5, 'K', 50, 'max_iter', 20);
```


 2. The following command specifies holdout evaluation, specified by the ratio of the testing set. Note that it is possible to specify the ratio of the training set by *train_ratio* (default=1), but it means that in can not be used separately. In other words, it should always be used together with *test_ratio*, since it means it will get the training set from the remaining part after excluding the test set. Besides, it should be accompanied by *times* option, indicating how many times are need for such holdout evaluation with specified test_ratio.
```matlab	
item_recommend(@(mat) iccf(mat, 'K', 50, 'max_iter', 20), data, 'test_ratio', 0.2, 'times', 5);
```	
 There are also five types of matrix split methods, specified by *split_mode*, the same as *fold_mode*.
  
 3. If you have both training set and test set, using the following commands for training and testing.
```matlab
item_recommend(@(mat) iccf(mat, 'alpha', 30, 'K', 50, 'max_iter', 20), train, 'test', test);
```
 4. If only dataset is provided, it will return topk recommendation results for each user, where *topk* is a parameter with default value 200.
	

Besides, it provides some easy-to-use utility function, including **hyperp_search** for searching parameters, **readContent** to read matrix from tuple-based dataset file/ feature file. You can refer script test/dataset/test_script.m for detailed usage.

Here I would like to emphasis the iccf algorithm[3], since it subsumes weighted regularized matrix factorization, and enables taking any feature from both user side and item side into account. Below I will introduce the usage of this algorithm. A lot of options, including dimension *K*, maximum number of iteration *max_iter*, the weight of positive preference *alpha*,  could be specified in this algorithm. In order to specify features from user side and item side, it can specify *X* as user side feature matrix and *Y* as item size feature matrix. Some examples are listed below
```matlab
item_recommend(@(mat) iccf(mat, 'K', 50, 'max_iter', 20, 'alpha', 30), data, 'test_ratio', 0.2, 'times', 5);
```
	
Note that in this case zeros(size(data,1), 0) is the default value of *X* and  zeros(size(data, 2), 0) is the default value of *Y*, and this is equivalent to WRMF[1].

When item/user features are fed, *reg_u* and *reg_i* is used to specify the weight/importance of these features.

```matlab
item_recommend(@(mat) iccf(mat, 'K', 50, 'max_iter', 20, 'alpha', 30, 'Y', item_featue_matrix, 'reg_i', 100), data, 'test_ratio', 0.2, 'times', 5); 
```
It is worth mentioning that **piccf** is a parallel version of **iccf**. You can specify the same options as **iccf**.

When using the iccf algorithm for recommendation from implicit feedback datasets, *alpha*, *reg_u* and *reg_i* are three important parameters affecting the recommendation performance. And these parameters may be re-tuned when changing the dimension of latent space.
	
1. Hu, Y., Koren, Y., & Volinsky, C. Collaborative filtering for implicit feedback datasets.  Proceedings of ICDM 2015 (pp. 263-272). IEEE.
2. Lian, D., Zhao, C., Xie, X., Sun, G., Chen, E., & Rui, Y. GeoMF: joint geographical modeling and matrix factorization for point-of-interest recommendation. Proceedings of SIGKDD 2014 (pp. 831-840). ACM.
3. Lian, D., Ge, Y., Zhang, F., Yuan, N. J., Xie, X., Zhou, T., & Rui, Y. Content-aware collaborative filtering for location recommendation based on human mobility data. Proceedings of ICDM 2015 (pp. 261-270). IEEE.
4. Liu, Y., Wei, W., Sun, A., & Miao, C. Exploiting geographical neighborhood characteristics for location recommendation. Proceedings of CIKM 2014 (pp. 739-748). ACM.


