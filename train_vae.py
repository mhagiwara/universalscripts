import argparse
import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_HIDDEN = 16
NUM_LAYER1 = 512

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(IMG_WIDTH * IMG_HEIGHT, NUM_LAYER1)
        self.fc21 = nn.Linear(NUM_LAYER1, NUM_HIDDEN)
        self.fc22 = nn.Linear(NUM_LAYER1, NUM_HIDDEN)
        self.fc3 = nn.Linear(NUM_HIDDEN, NUM_LAYER1)
        self.fc4 = nn.Linear(NUM_LAYER1, IMG_WIDTH * IMG_HEIGHT)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, IMG_WIDTH * IMG_HEIGHT))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    cross_entropy = F.binary_cross_entropy(
        recon_x, x.view(-1, IMG_WIDTH * IMG_HEIGHT), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_distance = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return cross_entropy + kl_distance


def generate(model, epoch):
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
            recon = model.decode(z)
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
                                    transforms.ToTensor()])
    chars = datasets.folder.ImageFolder('data/', transform=transform)
    train_loader = torch.utils.data.DataLoader(chars, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2.0e-3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

        model.eval()
        generate(model, epoch)

    torch.save(model.state_dict(), 'data/model.pt')


if __name__ == '__main__':
    main()
