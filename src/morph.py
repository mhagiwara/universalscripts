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


def main():
    model = VAE(image_channels=1, h_dim=IMG_WIDTH * IMG_HEIGHT, z_dim=NUM_HIDDEN)
    model.load_state_dict(torch.load('result/vae/model.pt'))

    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor()])
    image = load_image('test/2416.png', transform)
    z, _, _ = model.bottleneck(model.encoder(image))
    print(z)
    recon = model.decoder(model.fc3(z))
    save_image(recon, 'test/2416.recon.png')

if __name__ == '__main__':
    main()

