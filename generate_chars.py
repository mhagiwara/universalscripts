import numpy as np
import torch
from torchvision.utils import save_image

from train_vae import VAE, NUM_HIDDEN, IMG_WIDTH, IMG_HEIGHT


def load_model():
    model = VAE()
    model.load_state_dict(torch.load('data/model.pt'))
    model.eval()

    return model


def random():
    model = load_model()

    images = []
    for x in range(10+1):
        for y in range(10+1):

            z = torch.randn(NUM_HIDDEN).unsqueeze(0)
            recon = model.decode(z)
            images.append(recon)

    images_joined = torch.cat(images).view(-1, 1, IMG_WIDTH, IMG_HEIGHT)
    save_image(images_joined.cpu(),
               'data/reconstruction/random.png', nrow=11)


def morph():
    model = load_model()
    from torch.distributions.normal import Normal
    normal = Normal(0., 1.)

    images = []
    z = torch.randn(NUM_HIDDEN)
    for x in range(10+1):
        for y in range(10+1):
            x_coord = min(max(x / 10., .01), .99)
            y_coord = min(max(y / 10., .01), .99)

            z[0:2] = normal.icdf(torch.tensor([x_coord, y_coord]))
            recon = model.decode(z)
            images.append(recon)

    images_joined = torch.cat(images).view(-1, 1, IMG_WIDTH, IMG_HEIGHT)
    save_image(images_joined.cpu(),
               'data/reconstruction/morph.png', nrow=11)


def generate():
    model = load_model()

    for c in range(1000):
        filename = ''
        z = torch.zeros(NUM_HIDDEN)
        for d in range(NUM_HIDDEN):
            value = np.random.choice([1.2, -1.2], 1)
            if value == 1.2:
                filename += '1'
            else:
                filename += '0'
            z[d] = value[0]
        recon = model.decode(z).view(-1, 1, IMG_WIDTH, IMG_HEIGHT)
        save_image(recon,
                   'data/reconstruction/{}.png'.format(filename), nrow=1)


if __name__ == '__main__':
    morph()
