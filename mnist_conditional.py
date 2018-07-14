import torch
from christorch.gan.cgan import CGAN
from christorch.gan.architectures import (disc_conditional,
                                          gen_conditional)
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

mnist = MNIST('MNIST', download=True)
labels = torch.eye(10)[ mnist.train_labels ]
train_data = mnist.train_data.numpy()
train_data = ( (train_data/255.)-0.5)/0.5 # can't norm in torch, gives core dump exception???
train_data = torch.from_numpy(train_data).view(60000, 1, 28, 28)

ds = TensorDataset(train_data, labels)

batch_size = 32
data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

z_dim = 128
y_dim = 10
gan = CGAN(
    gen_fn=gen_conditional.generator(28, 28, 1, z_dim, y_dim),
    disc_fn=disc_conditional.discriminator(28, 28, 1, 1, y_dim,
                                           out_nonlinearity='sigmoid'),
    z_dim=z_dim,
    opt_d_args={'lr': 2e-4, 'betas': (0.5, 0.999)},
    opt_g_args={'lr': 2e-4, 'betas': (0.5, 0.999)},
    y_dim=y_dim,
    dnorm=10.
)

gan.train(itr=data_loader, epochs=200,
          model_dir=None, result_dir="tmp")
