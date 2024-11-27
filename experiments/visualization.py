import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        group_noise = f.readlines()

    epoch_list = []
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    for item in group_noise:
        item = item.strip().split()
        epoch_list.append(int(item[0]))
        train_acc.append(float(item[1]))
        train_loss.append(float(item[2]))
        test_acc.append(float(item[3]))
        test_loss.append(float(item[4]))
    return epoch_list, train_acc, test_acc, train_loss, test_loss

def plot_acc(epoch_list, train_acc, test_acc, title, save_path=None):
    plt.figure()
    plt.plot(epoch_list, train_acc, '-r*', label='Training acc')
    plt.plot(epoch_list, test_acc, '-b*', label='Testing acc')
    plt.xlabel('Epoch', size=13)
    plt.ylabel('Accuracy', size=13)
    plt.xticks(size=13)
    plt.yticks(size=13)

    x_major_locator = MultipleLocator(4)
    y_major_locator = MultipleLocator(0.2)

    ax = plt.gca()

    plt.xlim(0, 20)
    plt.ylim(0, 1.0)

    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # title = 'Training and Testing Accuracy (Original)'
    plt.title(title, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


def plot_loss(epoch_list, train_loss, test_loss, title, save_path=None):
    plt.figure()
    plt.plot(epoch_list, train_loss, '-r*', label='Training loss')
    plt.plot(epoch_list, test_loss, '-b*', label='Testing loss')
    plt.xlabel('Epoch', size=13)
    plt.ylabel('Loss', size=13)
    plt.xticks(size=13)
    plt.yticks(size=13)

    x_major_locator = MultipleLocator(4)
    # y_major_locator = MultipleLocator(0.002)
    y_major_locator = MultipleLocator(0.01)

    ax = plt.gca()

    plt.xlim(0, 21)
    # plt.ylim(0, 0.1)

    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.title(title, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    group_txt = r'D:\my_phd\on_git\Stage5\experiments\group_results.txt'
    group_noise_txt = r'D:\my_phd\on_git\Stage5\experiments\groupNoise_results.txt'

    # # Original
    # txt_path = group_txt
    # epoch_list, train_acc, test_acc, train_loss, test_loss = read_txt(txt_path)
    # plot_acc(epoch_list, train_acc, test_acc,
    #          title='Training and Testing Accuracy (Original)',
    #          save_path=r'D:\my_phd\on_git\Stage5\experiments\groupOrg_acc.png')
    # plot_loss(epoch_list, train_loss, test_loss,
    #           title='Training and Testing Loss (Original)',
    #           save_path=r'D:\my_phd\on_git\Stage5\experiments\groupOrg_loss.png')

    # Adding biases
    txt_path = group_noise_txt
    epoch_list, train_acc, test_acc, train_loss, test_loss = read_txt(txt_path)
    # plot_acc(epoch_list, train_acc, test_acc,
    #          title='Training and Testing Accuracy (Adding biases)',
    #          save_path=r'D:\my_phd\on_git\Stage5\experiments\groupNoise_acc.png')
    plot_loss(epoch_list, train_loss, test_loss,
              title='Training and Testing Loss (Adding biases)',
              save_path=r'D:\my_phd\on_git\Stage5\experiments\groupNoise_loss.png')







