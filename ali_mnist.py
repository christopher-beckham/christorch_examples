from christorch.gan.ali import ALI
from christorch.gan.architectures.ali import mnist as arch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=True
)

z_dim = 128
ali = ALI(
    gx=arch.GeneratorX(z_dim),
    gz=arch.GeneratorZ(z_dim),
    dx=arch.DiscriminatorX(z_dim),
    dxz=arch.DiscriminatorXZ(z_dim),
    z_dim=z_dim,
    opt_d_args={'lr': 1e-5, 'betas': (0.5, 0.999)},
    opt_g_args={'lr': 2e-4, 'betas': (0.5, 0.999)},

)


ali.train(itr=data_loader, epochs=200,
          model_dir=None, result_dir="tmp")

print(ali)
