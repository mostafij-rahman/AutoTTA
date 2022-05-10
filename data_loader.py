import numpy as np
from sklearn.model_selection import train_test_split

from keras import datasets
from keras.utils.np_utils import to_categorical


# datasets in the AutoAugment paper:
# CIFAR-10, CIFAR-100, SVHN, and ImageNet
# SVHN = http://ufldl.stanford.edu/housenumbers/


def get_dataset(dataset, reduced):
    if dataset == 'cifar10':
        (Xtr, ytr), (Xts, yts) = datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (Xtr, ytr), (Xts, yts) = datasets.cifar100.load_data()
    else:
        raise Exception('Unknown dataset %s' % dataset)
    if reduced:
        ix = np.random.choice(len(Xtr), 4000, False)
        Xtr = Xtr[ix]
        ytr = ytr[ix]
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.20, random_state=42)
    
    #np.save('data/Xtr.npy',Xtr)
    #np.save('data/Ytr.npy',ytr)
    #np.save('data/Xva.npy',Xva)
    #np.save('data/Yva.npy',yva)
    #np.save('data/Xts.npy',Xts)
    #np.save('data/Yts.npy',yts)
    Xtr = np.load('data/Xtr.npy')
    ytr = np.load('data/Ytr.npy')
    Xva = np.load('data/Xva.npy')
    yva = np.load('data/Yva.npy')
    Xts = np.load('data/Xts.npy')
    yts = np.load('data/Yts.npy')
    
    ytr = to_categorical(ytr, 10)
    yva = to_categorical(yva, 10)
    yts = to_categorical(yts, 10)
    return (Xtr, ytr), (Xva, yva), (Xts, yts)
