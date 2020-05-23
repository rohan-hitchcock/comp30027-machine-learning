import pandas as pd
import matplotlib.pyplot as plt

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


if __name__ == "__main__":

    kernals = ['linear', 'rbf', 'poly', 'sigmoid']
    ymetric = "fscore"



    fig = plt.figure()

    for i, kernal in enumerate(kernals, start=1):

        ax = fig.add_subplot(2, 2, i)

        mpl_plot_svm_learning_curve(ax, kernal, ymetric)


    
    plt.show()



    