import numpy as np
import torch
from torchvision.utils import save_image

from train_gan import Generator, NUM_HIDDEN, IMG_WIDTH, IMG_HEIGHT


def load_model():
    model = Generator(d=NUM_HIDDEN)
    model.load_state_dict(torch.load('data/generator092.pt'))
    model.eval()

    return model


def random():
    model = load_model()

    images = []
    for x in range(10+1):
        for y in range(10+1):

            z = torch.randn(NUM_HIDDEN).view(1, NUM_HIDDEN, 1, 1)
            recon = model(z)
            images.append(recon)

    images_joined = torch.cat(images).view(-1, 1, 2 * IMG_WIDTH, 2 * IMG_HEIGHT)
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
            value = np.random.choice([1., -1.], 1)
            if value == 1.:
                filename += '1'
            else:
                filename += '0'
            z[d] = value[0]
        z = z.view(1, NUM_HIDDEN, 1, 1)
        recon = model(z).view(-1, 1, 2 * IMG_WIDTH, 2 * IMG_HEIGHT)
        save_image(recon,
                   'data/reconstruction/{}.png'.format(filename), nrow=1)


if __name__ == '__main__':
    random()
