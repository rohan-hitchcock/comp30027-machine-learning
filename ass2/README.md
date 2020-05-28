### The University of Melbourne, School of Computing and Information Systems
# COMP30027 Machine Learning, 2020 Semester 1

## Assignment 2: This review sounds positive!

###### Submission deadline: 7 pm, Monday 20 Apr 2020

**Student Name(s):**    Patrick Randell, Rohan Hitchcock

**Student ID(s):**     836026, 836598

##### Questions Answered: Q1, Q2, Q4, Q5


## How to Generate all text features used in our code
The file `generate_docvecs.py` contains the functions used to generate train/test splits and cross validation splits of the original review text and train Doc2Vec text features on them. It has initialised a random seed that we used for generation. the two particular functions are:
* `compute_train_test_split(dim_start=25, dim_stop=300, dim_step=25)`
    * This was used to generate splits used for the learning curves of all models
* `compute_crossval_split(dim, num_splits)`
    * This was used for the optimal dimensions of each model (125, and 150)
    
## How to use relevant code
* ##### Feature Selection: `pca.py`
* ##### SVM: `svm.py`
    * PolarizedSvm class in `polarized_svm.py`
* ##### Logistic Regression: `mlr.py`
* ##### Stacking: `stacking.py`
* ##### Final shared plots: `plotting.py`
Each relevant file is commented explaining what particular functions are used for and how.
## Results
Results for each are found in the corresponding `results` folder. Figures and various other file types corresponding to data used to produce Figures are also in these folders.