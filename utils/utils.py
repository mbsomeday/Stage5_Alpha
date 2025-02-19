from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_cm(y_true, y_pred, label_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    print(f'cm:\n {cm}')
    # conf_matrix_df = pd.DataFrame(cm, columns=label_names, index=label_names)
    # sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    # plt.title(title)
    # plt.ylabel('Label')
    # plt.xlabel('Prediction')
    # plt.savefig(f'{title}.png')
    # plt.show()

















