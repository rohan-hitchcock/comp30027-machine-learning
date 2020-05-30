### The University of Melbourne, School of Computing and Information Systems
# COMP30027 Machine Learning, 2020 Semester 1

## Assignment 2: This review sounds positive!

###### Submission deadline: 11AM, Friday 29 May 2020

**Student Name(s):**    Patrick Randell, Rohan Hitchcock

**Student ID(s):**     836026, 836598


## How to Generate all text features used in our code
The file `generate_docvecs.py` contains the functions used to generate train/test splits and cross validation splits of the original review text and train Doc2Vec text features on them. It has initialised a random seed that we used for generation. the two particular functions are:
* `compute_train_test_split(dim_start=25, dim_stop=300, dim_step=25)`
    * This was used to generate splits used for the learning curves of all models
* `compute_crossval_split(dim, num_splits)`
    * This was used for the optimal dimensions of each model (125, and 150)
* functions within this file also load cross validation splits and random holdouts from file after they have been created

**NOTE:**
It is important that a folder `datasets` exists at the same level of all python files, containing the precomputed text features, and another folder `computed`. This computed folder is where all generated text features go (Each having their own generated folder).

    
## How to use relevant code
* ##### Feature Selection: `pca.py`
* ##### SVM: `svm.py`
    * This file contains a demonstration of how to run each model on the whole training set
    * PolarizedSvm class in `polarized_svm.py` and imported into this file. In our report PolarizedSVM was renamed to Binary-SVM since this is more clear.
* ##### Logistic Regression: `mlr.py`
* ##### Stacking: `stacking.py`
* ##### Final shared plots: `plotting.py`
Each relevant file is commented explaining what particular functions are used for and how.
## Results
Results for each are found in the corresponding `results` folder. Figures and various other file types corresponding to data used to produce Figures are also in these folders.