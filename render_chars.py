from PIL import Image, ImageFont, ImageDraw

IMG_WIDTH = 32
IMG_HEIGHt = 32

def render_char(char):
    image = Image.new("RGBA", (32, 32), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("data/unifont-12.0.01.ttf", size=24)
    width, height = font.getsize(char)

    draw.text(((IMG_WIDTH - width) / 2, (IMG_HEIGHt - height) / 2), char, (255, 255, 255), font=font)
    image.save('data/chars/u{:04x}.png'.format(ord(char)))


def main():
    with open('data/sampled_chars.txt') as f:
        for line in f:
            char = line.strip()
            render_char(char)


if __name__ == '__main__':
    main()
