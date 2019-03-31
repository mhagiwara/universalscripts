import argparse

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_HIDDEN = 16
NUM_LAYER1 = 512

class Generator(nn.Module):
    def __init__(self, input_size, n_class):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(input_size, NUM_LAYER1)
        self.fc2 = nn.Linear(self.fc1.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.tanh(self.fc2(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, n_class):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_size, NUM_LAYER1)
        self.fc2 = nn.Linear(self.fc1.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.sigmoid(self.fc2(x))

        return x


def generate(generator, epoch):
    from torch.distributions.normal import Normal
    images = []
    normal = Normal(0., 1.)
    for x in range(10+1):
        for y in range(10+1):
            x_coord = min(max(x / 10., .01), .99)
            y_coord = min(max(y / 10., .01), .99)

            z = torch.randn(NUM_HIDDEN).unsqueeze(0)
            # z = torch.zeros(NUM_HIDDEN)
            # z[0:2] = normal.icdf(torch.tensor([x_coord, y_coord]))
            recon = generator(z)
            images.append(recon)

    images_joined = torch.cat(images).view(-1, 1, IMG_WIDTH, IMG_HEIGHT)
    save_image(images_joined.cpu(),
               'data/reconstruction/generated{:02d}.png'.format(epoch), nrow=11)


def main():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.,), std=(1.,))])
    chars = datasets.folder.ImageFolder('data/', transform=transform)
    train_loader = torch.utils.data.DataLoader(chars, batch_size=args.batch_size, shuffle=True)

    generator = Generator(input_size=NUM_HIDDEN, n_class=IMG_WIDTH * IMG_HEIGHT)
    discriminator = Discriminator(input_size=IMG_WIDTH * IMG_HEIGHT, n_class=1)

    lr = 1.0e-3
    optim_g = optim.Adam(generator.parameters(), lr=lr)
    optim_d = optim.Adam(discriminator.parameters(), lr=lr)

    bce_loss = nn.BCELoss()

    for epoch in range(args.epochs):
        losses_d = []
        losses_g = []

        for x, _ in train_loader:
            # train discriminator
            discriminator.zero_grad()

            x = x.view(-1, IMG_HEIGHT * IMG_WIDTH)
            batch_size = x.size()[0]

            real_target = torch.ones(batch_size)
            fake_target = torch.zeros(batch_size)

            real_pred = discriminator(x)
            real_loss = bce_loss(real_pred, real_target)

            z = torch.randn((batch_size, NUM_HIDDEN))
            generated = generator(z)

            fake_pred = discriminator(generated)
            fake_loss = bce_loss(fake_pred, fake_target)
            train_loss = real_loss + fake_loss

            train_loss.backward()
            optim_d.step()

            losses_d.append(train_loss)

            # train generator
            generator.zero_grad()

            # z = torch.randn((args.batch_size, NUM_HIDDEN))
            target = torch.ones(batch_size)

            generated = generator(z)
            pred = discriminator(generated)
            train_loss = bce_loss(pred, target)
            train_loss.backward()
            optim_g.step()

            losses_g.append(train_loss)

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), args.epochs, torch.mean(torch.tensor(losses_d)), torch.mean(torch.tensor(losses_g))))

        generate(generator, epoch)


if __name__ == '__main__':
    main()