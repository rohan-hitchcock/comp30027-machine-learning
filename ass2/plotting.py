import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

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

    sns.heatmap(heat_map, vmin=vmin, vmax=vmax, ax=ax, cmap="YlGnBu", xticklabels=xax_vals, yticklabels=yax_vals)

def gridsearch_heatmap_1d(ax, df, xax_col, val_col, lims=None):

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

    xax_vals = [round(xv, ndigits=3) for xv in xax_vals]
    heat_map = heat_map.transpose()

    sns.heatmap(heat_map, vmin=vmin, vmax=vmax, ax=ax, cmap="YlGnBu", xticklabels=False, yticklabels=xax_vals, square=True)

if __name__ == "__main__":

    gridsearch = pd.read_csv("./results/svm/gridsearch_linear_150.csv", sep=', ')
    

    

    #gridsearch_rbf = gridsearch_rbf[(gridsearch_rbf['C'] <= 1.75) & (gridsearch_rbf['C'] >= 0.5)]


    xax_col = 'C'
    val_col = 'fscore'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fig.tight_layout()

    
    gridsearch_heatmap_1d(ax, gridsearch, xax_col, val_col)

    #ax.set_xlabel("C")

    plt.show()


    