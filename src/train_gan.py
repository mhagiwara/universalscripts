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


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(d, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


z_fixed = [[torch.randn(NUM_HIDDEN).view(1, NUM_HIDDEN, 1, 1) for y in range(10+1)]
           for x in range(10+1)]

def generate_random(generator, epoch):
    from torch.distributions.normal import Normal
    images = []
    for x in range(10+1):
        for y in range(10+1):

            z = torch.randn(NUM_HIDDEN).view(1, NUM_HIDDEN, 1, 1)
            recon = generator(z)
            images.append(recon)

    images_joined = torch.cat(images).view(-1, 1, 2 * IMG_WIDTH, 2 * IMG_HEIGHT)
    save_image(images_joined.cpu(),
               'result/gan/epoch{:03d}.png'.format(epoch), nrow=11)


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

    transform = transforms.Compose([transforms.Scale(64),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(.5,), std=(.5,))])
    chars = datasets.folder.ImageFolder('data/', transform=transform)
    train_loader = torch.utils.data.DataLoader(chars, batch_size=args.batch_size, shuffle=True)

    generator = Generator(d=NUM_HIDDEN)
    discriminator = Discriminator(d=NUM_HIDDEN)

    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)

    lr = 2.0e-3
    optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(.5, .999))
    optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(.5, .999))

    bce_loss = nn.BCELoss()

    for epoch in range(args.epochs):
        generator.train()

        losses_d = []
        losses_g = []

        for x, _ in train_loader:
            # train discriminator
            discriminator.zero_grad()

            batch_size = x.size()[0]

            real_target = torch.ones(batch_size)
            fake_target = torch.zeros(batch_size)

            real_pred = discriminator(x)
            real_loss = bce_loss(real_pred, real_target)

            z = torch.randn((batch_size, NUM_HIDDEN, 1, 1))
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

        generate_random(generator, epoch)

        # torch.save(generator.state_dict(), 'data/generator{:03d}.pt'.format(epoch))


if __name__ == '__main__':
    main()