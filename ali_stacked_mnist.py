import torch
from christorch.gan.ali import ALI
from christorch.gan.architectures.ali import mnist as arch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class MNISTStacked(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(MNISTStacked, self).__init__(*args, **kwargs)
        self.rnd_state = np.random.RandomState(0)
    def __getitem__(self, index):
        N = self.train_data.shape[0]
        x, _ = super(MNISTStacked, self).__getitem__(index)
        # Ok, now randomly grab two other images.
        x2, _ = super(MNISTStacked, self).__getitem__(self.rnd_state.randint(0,N))
        x3, _ = super(MNISTStacked, self).__getitem__(self.rnd_state.randint(0,N))
           
        return torch.cat((x,x2,x3), dim=0)

batch_size = 32
data_loader = DataLoader(
    MNISTStacked('MNIST', download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=True
)

z_dim = 128
ali = ALI(
    gx=arch.GeneratorX(z_dim, ch=3),
    gz=arch.GeneratorZ(z_dim, ch=3),
    dx=arch.DiscriminatorX(z_dim, ch=3),
    dxz=arch.DiscriminatorXZ(z_dim),
    z_dim=z_dim,
    lamb=0.1,
    opt_d_args={'lr': 1e-5, 'betas': (0.5, 0.999)},
    opt_g_args={'lr': 2e-4, 'betas': (0.5, 0.999)},

)

ali.train(itr=data_loader, epochs=200,
          model_dir=None, result_dir="stacked2",
          save_every=10)

#ali.load("stacked_mnist/146.pkl.bak")

import pdb
pdb.set_trace()

ali.sample(100)

            
print(ali)
