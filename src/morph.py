from src.train_vae import VAE, IMG_WIDTH, IMG_HEIGHT, NUM_HIDDEN
from torchvision import datasets, transforms
from PIL import Image
import torch
from torchvision.utils import save_image


def load_image(file_path, transform):
    image = Image.open(file_path)
    image = transform(image).float()
    image = torch.Tensor(image)
    image = image.unsqueeze(0)
    return image


def reconstruct():
    model = VAE(image_channels=1, h_dim=IMG_WIDTH * IMG_HEIGHT, z_dim=NUM_HIDDEN)
    model.load_state_dict(torch.load('result/vae/model.pt'))
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor()])

    image = load_image('test/2416.png', transform)
    z, _, _ = model.bottleneck(model.encoder(image))
    print(z)
    recon = model.decoder(model.fc3(z))
    save_image(recon, 'test/2416.recon.png')


def morph():
    model = VAE(image_channels=1, h_dim=IMG_WIDTH * IMG_HEIGHT, z_dim=NUM_HIDDEN)
    model.load_state_dict(torch.load('result/vae/model.pt'))
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor()])

    image1 = load_image('test/traditional_fan.png', transform)
    z1, _, _ = model.bottleneck(model.encoder(image1))

    image2 = load_image('test/simplified_fan.png', transform)
    z2, _, _ = model.bottleneck(model.encoder(image2))

    steps = 10
    diff = (z2 - z1) / 10
    images = [image1]

    z = z1
    for i in range(steps):
        recon = model.decoder(model.fc3(z))
        images.append(recon)

        z += diff

    images.append(image2)

    for i in range(steps):
        recon = model.decoder(model.fc3(z))
        images.append(recon)

        z += diff

    images_joined = torch.cat(images).view(-1, 1, IMG_WIDTH, IMG_HEIGHT)
    save_image(images_joined,
               'test/morph.png', nrow=22)


if __name__ == '__main__':
    morph()

