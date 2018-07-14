from christorch.gan.base import GAN
from christorch.gan.architectures import disc, gen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

class MNISTNoClass(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(MNISTNoClass, self).__init__(*args, **kwargs)
    def __getitem__(self, index):
        x, _ = super(MNISTNoClass, self).__getitem__(index)
        return x

batch_size = 32
data_loader = DataLoader(
    MNISTNoClass('MNIST', download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                   ])),
    batch_size=batch_size, shuffle=True
)

z_dim = 128
gan = GAN(
    gen_fn=gen.generator(28, 28, 1, z_dim),
    disc_fn=disc.discriminator(28, 28, 1, 1,
                               out_nonlinearity='sigmoid'),
    z_dim=z_dim,
    opt_d_args={'lr': 2e-4, 'betas': (0.5, 0.999)},
    opt_g_args={'lr': 2e-4, 'betas': (0.5, 0.999)},
    dnorm=10.
)

gan.train(itr=data_loader, epochs=200,
          model_dir=None, result_dir="tmp")

