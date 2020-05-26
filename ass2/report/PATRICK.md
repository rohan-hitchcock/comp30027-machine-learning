### **DATA EXPLORATION**

#### Word Occurrences and Correlations
We were interested to know which words from the vocabulary of the entire dataset were most predictive of a class label alone, and if these words would be intuitively expected. The graph in figure (/results/lgr/figures/CountVEC_KBest_Occurences) shows the 20 best CountVectoriser features (by Chi^2) score, and their occurrences across all reviews. Chi^2 was chosen for this task as it is a normalised value; therefore, these values are more comparable across terms in the same category.
(Aggarwal Charu C, Zhai Cheng Xiang. Mining Text Data. Springer New York Dordrecht Heidelberg London: Ó Springer Science+Business Media, LLC’12; 2012 _[62] from the review_) . Note that this graph does not show the class label each feature is correlated to. The graph shows some expected, and some more unexpected results. It is perhaps unsurprising that “rude” has a high score, and we can guess which rating this word is correlated most strongly with. However, words such as “asked” are less obvious. One might suspect that a reviewer would speak positively of a restaurant if their waiter or waitress continually asked if the customer needed something, or if their meal was to their satisfaction. On the other hand, a customer may have asked for something of the waiter or waitress, that wasn’t delivered or refused. The extremely high frequency of the word “good” can likely be attributed to its many uses, such as “not good”, “pretty good” or just “good”.

#### Clustering and Component Reduction
As an early attempt to visualise the data, to see if there where any clearly visible clusters or relationships in the data, we used Principle Component Analysis on the generated text features, with 2 and 3 Principle Components, giving different colours to each of the class labels. For CountVectoriser we used TruncatedSVD. TruncatedSVD is similar to PCA, except that it does not centre supplied matrix, making it suitable for sparse input (https://nlp.stanford.edu/IR-book/pdf/18lsi.pdf). From the figure below, it can clearly be seen that even with as little as 2 components, there are visible clusters representing the different class labels in the text features, although there is significant overlap between the classes.
(results/pca/DimensionReduction)

We wanted to explore further into the amount of variance captured by the principle components. 
The following graphs show the total variance explained for an increasing number of principle components, compared for  when data is Standardised before dimension reduction and when it is not. (results/pca/NormalisedvsUnnormalised) Standardising is usually done when the variables on which the PCA (or TSVD) is performed are not measured on the same scale. It implies assigning equal importances to all variables. The graphs show that for Doc2Vec, the increased dimensionality does in fact help in distinguishing each document. Doc2Vec200 with 50 principle components does not capture the amount of variance as doc2Vec50 for the same number of principle components, implying that the remaining dimensions carry important information used to distinguish documents that cannot be simplified by further component reduction. We also notice that normalisation of Doc2Vec text features makes practically no difference to the explained variance. This makes sense, as each “feature” is a dimension in which to seperate each document. They are all on the same scale and equally important. However, for CountVectoriser, features correspond to words and their frequencies. Despite being on the same scale, they are not distributed normally. This may explain why normalisation considerably reduced the explained variance for CountVectoriser. Due to a having over 46000 features, corresponding to the combined vocabulary of the dataset, a much larger number of components might be needed to capture more variance in the entire dataset. Words with high occurences may have been favoured in the unstandardised graph.

To complete the analysis of Dimension Reduction of Doc2Vec features, we compared the results using Logistic Regression (Motivation for which is explained in Logistic Regression section) for Higher dimensionality Doc2Vec text features (100 and 200), reduced using PCA to 50 Principle components, to Doc2Vec50 itself. We used Stratified Cross Validation with 10 splits, 10 times each with different random seeds, training the Principle Components on the train partitions and transforming the test partitions with the fitted PCA models. It can be seen from the results in Figure___ (results/lgr/figures/Dimensionality_Reduction) that Doc2Vec50 outperforms the reduced higher dimensional Doc2Vec text features. Thus we can conclude that higher dimensional Doc2Vec text features should not be further reduced in dimensionality if they are to be used, as information gained is lost in the reduction.

### **LOGISITIC REGRESSION**

Based on (Sentiment analysis survey) Comparing different techniques for sentiment classification, it was suggested that Maximum Entropy (Identical to “Logistic Regression” as called in lectures) may perform better as a probabilistic supervised Machine learning classifier than the more commonly used Niave Bayes, as it does not make the assumption that features are conditionally independent. Niave Bayers has higher bias but lower variance than Logisitic Regression, and in general, as the training size tends to infinity, Logistic Regression performs better as a classifier (Summarised https://medium.com/@sangha_deb/naive-bayes-vs-logistic-regression-a319b07a5d4c) 

We wanted to re-generate Doc2Vec features to determine the ideal dimensionality, and have them generated on each partition of a Stratified Cross Validation to prevent overfitting. This took some time, and so some of the preliminary analysis of Logistic Regression as a model for this task was done with the pre-computed Doc2Vec text features.

#### TEXT FEATURE SELECTION
To determine whether CountVectoriser could be used in addition to any Doc2Vec text feature to improve classification performance, we added the K-Best CountVectoriser features to Doc2Vec50 and compared it’s performance compared to the K-best CountVectoriser features alone. The graph in figure ____ (results/lgr/figures/CountVec_vs_Doc2Vec) shows the addition of the first 30 best CountVectoriser features to Doc2Vec50, and the first 100 best features alone. This was due to time-constraints, and the results are clear without generating more features. Doc2Vec50 performs significantly better than CountVectoriser alone, and improves only slightly with the addition of CountVectoriser’s best features.
Doc2Vec text features build a Bag-Of-Words model and use Neural Networks to predict the next words in context, and thus we would suspect that performance improvement may be mostly due to over-fitting. The K-best words from CountVectoriser were calculated prior to cross-validation, and they therefore come from data unseen to the test sets of each partition.

#### C HYPERPARAMETER
Show graph yada yada (results/lgr/figures/LogisReg_vs_C). Stock C parameter is best.

#### LEARNING CURVE
150 ideal (results/lgr/figures/LGR_LearningCurve_D2V)

#### ENSEMBLING
Upon initial inspection, bagging looks promising (results/lgr/figures/EnsembleCompare)
On further inspection, averaged over multiple random seeds, it is not (results/lgr/figures/BaggingCompare)

#### SUMMARY
LGR with C=1, Doc2Vec150, no added features seems to be ideal?











Chi^2 is better than PMI as it is a normalised value; therefore, these values are more comparable across terms in the same category

Aggarwal Charu C, Zhai Cheng Xiang. Mining Text Data. Springer New York Dordrecht Heidelberg London: Ó Springer Science+Business Media, LLC’12; 2012. 
