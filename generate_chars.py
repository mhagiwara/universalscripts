import numpy as np
import torch
from torchvision.utils import save_image

from train_vae import VAE, NUM_HIDDEN, IMG_WIDTH, IMG_HEIGHT


def main():
    model = VAE()
    model.load_state_dict(torch.load('data/model.pt'))
    model.eval()

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
    main()
