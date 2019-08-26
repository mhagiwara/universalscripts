"""Script to split chars into individual files."""

from PIL import Image
import PIL.ImageOps

def main():
    image = Image.open('data/chars.png')
    r, g, b, a = image.split()
    image = Image.merge('RGB', (r, g, b))

    image = PIL.ImageOps.invert(image)

    x_offset = 8
    y_offset = 6
    for x in range(40):
        for y in range(21):
            box = (x_offset + x * 48, y_offset + y * 48,
                   x_offset + x * 48 + 32, y_offset + y * 48 + 32)
            region = image.crop(box)
            region.save('data/chars/{:02d}{:02d}.png'.format(x, y))

if __name__ == '__main__':
    main()
