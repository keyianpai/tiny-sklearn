# tiny-sklearn

## About
- Tiny implementation of important algorithms in scikit-learn
(e.g., pure python, no input validation, no speed/memory optimization, do not support sparse matrix and multioutput).
- Useful when understanding ML algorithms and scikit-learn.
- Multiple implementations of each algorithm.
- Roughly follow the structure of scikit-learn.
- Roughly follow the API standard of scikit-learn.
- Results are compared with scikit-learn.

## Table of Contents
- **calibration** (sklearn.calibration)
  * [calibration_curve](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/calibration/calibration_curve.ipynb)
  * [CalibratedClassifierCV](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/calibration/CalibratedClassifierCV.ipynb)
- **cluster** (sklearn.cluster)
  * [AgglomerativeClustering](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/cluster/AgglomerativeClustering.ipynb)
  * [DBSCAN](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/cluster/DBSCAN.ipynb)
  * [KMeans](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/cluster/KMeans.ipynb)
- **covariance** (sklearn.covariance)
  * [EmpiricalCovariance](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/covariance/EmpiricalCovariance.ipynb)
- **decomposition** (sklearn.decomposition)
  * [KernelPCA](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/decomposition/KernelPCA.ipynb)
  * [PCA](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/decomposition/PCA.ipynb)
  * [TruncatedSVD](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/decomposition/TruncatedSVD.ipynb)
- **discriminant_analysis** (sklearn.discriminant_analysis)
  * [LinearDiscriminantAnalysis](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/discriminant_analysis/LinearDiscriminantAnalysis.ipynb)
- **dummy** (sklearn.dummy)
  * [DummyClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/dummy/DummyClassifier.ipynb)
  * [DummyRegressor](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/dummy/DummyRegressor.ipynb)
- ensemble (sklearn.ensemble)
  * [AdaBoostClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/ensemble/AdaBoostClassifier.ipynb)
  * [AdaBoostRegressor](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/ensemble/AdaBoostRegressor.ipynb)
  * [BaggingClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/ensemble/BaggingClassifier.ipynb)
  * [BaggingRegressor](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/ensemble/BaggingRegressor.ipynb)
  * [RandomForestClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/ensemble/RandomForestClassifier.ipynb)
  * [RandomForestRegressor](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/ensemble/RandomForestRegressor.ipynb)
  * [VotingClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/ensemble/VotingClassifier.ipynb)
  * [VotingRegressor](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/ensemble/VotingRegressor.ipynb)
- **feature_extraction** (sklearn.feature_extraction)
  * [CountVectorizer](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/feature_extraction/CountVectorizer.ipynb)
  * [TfidfTransformer](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/feature_extraction/TfidfTransformer.ipynb)
  * [TfidfVectorizer](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/feature_extraction/TfidfVectorizer.ipynb)
- feature_selection (sklearn.feature_selection)
- **impute** (sklearn.inpute)
  * [MissingIndicator](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/impute/MissingIndicator.ipynb)
  * [SimpleImputer](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/impute/SimpleImputer.ipynb)
- **inspection** (sklearn.inspection)
  * [partial_dependence](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/inspection/partial_dependence.ipynb)
  * [permutation_importance](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/inspection/permutation_importance.ipynb)
- **kernel_ridge** (sklearn.kernel_ridge)
  * [KernelRidge](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/kernel_ridge/KernelRidge.ipynb)
- **linear_model** (sklearn.linear_model)
  * [LinearRegression](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/linear_model/LinearRegression.ipynb)
  * [LogisticRegression](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/linear_model/LogisticRegression.ipynb)
  * [Ridge](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/linear_model/Ridge.ipynb)
  * [RidgeClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/linear_model/RidgeClassifier.ipynb)
  * [SGDClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/linear_model/SGDClassifier.ipynb) 
  * [SGDRegressor](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/linear_model/SGDRegressor.ipynb) 
- metrics (sklearn.metrics) - classification metrics
  * [accuracy_score](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/accuracy_score.ipynb)
  * [brier_score_loss](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/brier_score_loss.ipynb)
  * [confusion_matrix](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/confusion_matrix.ipynb)
  * [f1_score](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/f1_score.ipynb)
  * [fbeta_score](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/fbeta_score.ipynb)
  * [multilabel_confusion_matrix](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/multilabel_confusion_matrix.ipynb)
  * [precision_score](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/precision_score.ipynb)
  * [recall_score](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/recall_score.ipynb)
- **metrics** (sklearn.metrics) - regression metrics
  * [mean_absolute_error](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/mean_absolute_error.ipynb)
  * [mean_squared_error](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/mean_squared_error.ipynb)
  * [median_absolute_error](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/median_absolute_error.ipynb)
  * [r2_score](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/r2_score.ipynb)
- **metrics** (sklearn.metrics) - pairwise metrics
  * [cosine_distances](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/cosine_distances.ipynb)
  * [cosine_similarity](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/cosine_similarity.ipynb)
  * [euclidean_distances](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/euclidean_distances.ipynb)
  * [linear_kernel](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/linear_kernel.ipynb)
  * [rbf_kernel](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/metrics/rbf_kernel.ipynb)
- **mixture** (sklearn.mixture)
  * [GaussianMixture](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/mixture/GaussianMixture.ipynb)
- **neighbors** (sklearn.neighbors)
  * [KNeighborsClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/neighbors/KNeighborsClassifier.ipynb)
  * [KNeighborsRegressor](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/neighbors/KNeighborsRegressor.ipynb)
  * [NearestCentroid](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/neighbors/NearestCentroid.ipynb)
  * [NearestNeighbors](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/neighbors/NearestNeighbors.ipynb)
  * [RadiusNeighborsClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/neighbors/RadiusNeighborsClassifier.ipynb)
  * [RadiusNeighborsRegressor](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/neighbors/RadiusNeighborsRegressor.ipynb)
- **model_selection** (sklearn.model_selection) - splitter classes / functions
  * [KFold](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/KFold.ipynb)
  * [ShuffleSplit](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/ShuffleSplit.ipynb)
  * [StratifiedKFold](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/StratifiedKFold.ipynb)
  * [StratifiedShuffleSplit](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/StratifiedShuffleSplit.ipynb)
  * [TimeSeriesSplit](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/TimeSeriesSplit.ipynb)
  * [train_test_split](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/train_test_split.ipynb)
- model_selection (sklearn.model_selection) - hyper-parameter optimizers
- **model_selection** (sklearn.model_selection) - model validation
  * [cross_val_predict](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/cross_val_predict.ipynb)
  * [cross_val_score](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/cross_val_score.ipynb)
  * [learning_curve](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/learning_curve.ipynb)
  * [validation_curve](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/model_selection/validation_curve.ipynb)
- **multiclass** (sklearn.multiclass)
  * [OneVsOneClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/multiclass/OneVsOneClassifier.ipynb)
  * [OneVsRestClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/multiclass/OneVsRestClassifier.ipynb)
  * [OutputCodeClassifier](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/multiclass/OutputCodeClassifier.ipynb)
- **naive_bayes** (sklearn.naive_bayes)
  * [BernoulliNB](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/naive_bayes/BernoulliNB.ipynb)
  * [ComplementNB](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/naive_bayes/ComplementNB.ipynb)
  * [GaussianNB](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/naive_bayes/GaussianNB.ipynb)
  * [MultinomialNB](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/naive_bayes/MultinomialNB.ipynb)
- neural_network (sklearn.neural_network)
- **preprocessing** (sklearn.preprocessing)
  * [Binarizer](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/Binarizer.ipynb)
  * [KBinsDiscretizer](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/KBinsDiscretizer.ipynb)
  * [KernelCenterer](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/KernelCenterer.ipynb)
  * [LabelBinarizer](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/LabelBinarizer.ipynb)
  * [LabelEncoder](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/LabelEncoder.ipynb)
  * [MaxAbsScaler](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/MaxAbsScaler.ipynb)
  * [MinMaxScaler](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/MinMaxScaler.ipynb)
  * [Normalizer](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/Normalizer.ipynb)
  * [RobustScaler](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/RobustScaler.ipynb)
  * [StandardScaler](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/preprocessing/StandardScaler.ipynb)
- svm (sklearn.svm)
  * [LinearSVC](https://nbviewer.jupyter.org/github/qinhanmin2014/tiny-sklearn/blob/master/svm/LinearSVC.ipynb)
- tree (sklearn.tree)
