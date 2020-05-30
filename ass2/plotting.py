import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns;
from sklearn.metrics import ConfusionMatrixDisplay
sns.set()

def captialize(word):
    return word[0].upper() + word[1:]

def mpl_plot_svm_learning_curve(ax, name, ymetric):

    test_file_name = f"./results/svm/learning_curve_{name}_test.csv"
    train_file_name = f"./results/svm/learning_curve_{name}_train.csv"

    df_test = pd.read_csv(test_file_name, index_col=0, header=0)
    df_train = pd.read_csv(train_file_name, index_col=0, header=0)

    ax.plot(df_test['dim'], df_test[ymetric], 'go-', label="Test")
    ax.plot(df_train['dim'], df_train[ymetric], 'bo-', label="Train")

    ax.legend(loc='lower right', title="Dataset", title_fontsize=12)

    ax.set_title(f"Learning curve for {name} SVM")
    ax.set_xlabel("Feature dimension")
    ax.set_ylabel(f"{captialize(ymetric)}")

    ax.set_ylim(bottom=0.68, top=0.95)

def gridsearch_heatmap(ax, df, xax_col, yax_col, val_col, lims=None):

    xax_vals = np.unique(df[xax_col])
    yax_vals = np.unique(df[yax_col])

    heat_map = np.empty((len(yax_vals), len(xax_vals)))

    for xi, xval in enumerate(xax_vals):

        for yi, yval in enumerate(yax_vals):

            row = df[(df[xax_col] == xval) & (df[yax_col] == yval)]

            if row.empty:
                heat_map[yi][xi] = np.nan

            elif len(row) == 1:
                heat_map[yi][xi] = float(row[val_col])

            else:
                print("WARNING: multiple values encountered gridsearch:")
                print(row)
                print("Filling with nan.")
                heat_map[yi][xi] = np.nan
    
    vmin, vmax = lims if lims is not None else (None, None)
    xax_vals = [str(xv *1000) for xv in xax_vals]
    sns.heatmap(heat_map, vmin=vmin, vmax=vmax, ax=ax, cmap="YlGnBu", xticklabels=xax_vals, yticklabels=yax_vals)

def gridsearch_heatmap_1d(ax, df, xax_col, val_col, lims=None):


    print(df)

    xax_vals = np.unique(df[xax_col])

    heat_map = np.empty(len(xax_vals))

    for xi, xval in enumerate(xax_vals):


        row = df[(df[xax_col] == xval)]

        if row.empty:
            heat_map[xi] = np.nan

        elif len(row) == 1:
            heat_map[xi] = float(row[val_col])

        else:
            print("WARNING: multiple values encountered gridsearch:")
            print(row)
            print("Filling with nan.")
            heat_map[xi] = np.nan

    vmin, vmax = lims if lims is not None else (None, None)

    heat_map = np.expand_dims(heat_map, axis=0)

    xax_vals = [round(xv, ndigits=4) for xv in xax_vals]
    heat_map = heat_map.transpose()

    sns.heatmap(heat_map, vmin=vmin, vmax=vmax, ax=ax, cmap="YlGnBu", xticklabels=False, yticklabels=xax_vals, square=True)
def plot_confusion_matrix(cm):

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '3', '5'])
    disp.plot(include_values='true',
              cmap=plt.cm.Blues)
    plt.grid(b=False)

    plt.ylabel('True Rating')
    plt.xlabel('Predicted Rating')
    plt.tight_layout()
    plt.show()

def row_normalise(a):

    row_sum = a.sum(axis=1)
    return (a.transpose() / row_sum).transpose()

if __name__ == "__main__":

    color_map = cm.get_cmap("YlGnBu")

    colors = color_map(np.linspace(0.4, 1, 4))

    colors = ['c', 'g', 'r', 'b']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    learning_curves = [("./results/svm/learning_curve_binary_final_train.csv", "./results/svm/learning_curve_binary_final_test.csv"), 
     ("./results/svm/learning_curve_linear_final_train.csv", "./results/svm/learning_curve_linear_final_test.csv"),
     ("./results/svm/learning_curve_rbf_final_train.csv", "./results/svm/learning_curve_rbf_final_test.csv"), 
     ("./results/lgr/learning_curve_train.csv", "./results/lgr/learning_curve_test.csv")]
    
    
    names = ["Binary-SVM", "Linear-SVM", "RBF-SVM", "LR"]

    for color, files, name in zip(colors, learning_curves, names):

        if name == "Binary-SVM":
            continue

        train_file, test_file = files
        
        sep = "," if name == "LR" else ", "

        train_lc = pd.read_csv(train_file, sep=sep)
        test_lc = pd.read_csv(test_file, sep=sep)


        ax.plot(train_lc["dim"], train_lc["fscore"], color=color, linestyle="--", label=f"{name} Train", marker=".")
        ax.plot(test_lc["dim"], test_lc["fscore"], color=color, linestyle="-", label=f"{name} Test", marker=".")
    
    ax.legend(loc='best', fontsize="x-small")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("F-Score")
    ax.set_ylim(bottom=0.75, top=1)
    ax.set_xlim(left=0, right=300)
    ax.set_facecolor('w')

    fig.tight_layout()

    plt.show()

    