'''
=============================================================
Created and Commented by :
Farid Muhammad Eryanto - 1103181221

These Codes are reproduced from following github repository :
https://github.com/ansh941/MnistSimpleCNN

These repository was created as final exam task for 
Telkom University's Machine Learning G5 Subject
=============================================================
'''

# Import Required Module(s)
import torch # pytorch is used for deep neural network stuff
import numpy as np # for math stuff
from PIL import Image #for image processing stuff
from torchvision import transforms
# The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, training=True, transform=None):
        if training==True:
            f = open('../data/MNIST/raw/train-images-idx3-ubyte', 'rb')
            xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
            f.close()
            f = open('../data/MNIST/raw/train-labels-idx1-ubyte', 'rb')
            ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
            f.close()
        else:
            f = open('../data/MNIST/raw/t10k-images-idx3-ubyte', 'rb')
            xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
            f.close()
            f = open('../data/MNIST/raw/t10k-labels-idx1-ubyte', 'rb')
            ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
            f.close()
        # Reshape using numpy
        xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float32)
        ys = ys.astype(np.int)
        self.x_data = xs
        self.y_data = ys
        self.transform = transform

    # Return the length of the data
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = Image.fromarray(self.x_data[idx].reshape(28, 28))
        y = torch.tensor(np.array(self.y_data[idx]))
        if self.transform:
            x = self.transform(x)
        x = transforms.ToTensor()(np.array(x)/255)
        return x, y

