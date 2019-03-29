from PIL import Image, ImageFont, ImageDraw

IMG_WIDTH = 32
IMG_HEIGHT = 32

def render_char(char):
    image = Image.new("RGBA", (32, 32), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("data/unifont-12.0.01.ttf", size=24)
    width, height = font.getsize(char)

    draw.text(((IMG_WIDTH - width) / 2, (IMG_HEIGHT - height) / 2), char, (255, 255, 255), font=font)
    image.save('data/chars/u{:04x}.png'.format(ord(char)))


def create_html(chars):
    print("""
<html>
  <head>
    <link href="https://fonts.googleapis.com/css?family=Noto+Sans" rel="stylesheet">
    <style>
      body {
        font-family: 'Noto Sans', sans-serif;
        padding: 0;
        margin: 0;
      }
      table {
        margin: 0;
        padding: 0;
        font-size: 28px;
        border-collapse: collapse;
        table-layout: fixed;
      }
      tr {
        padding: 0;
        margin: 0;
      }
      td {
        width: 48px;
        height: 48px;
        padding: 0;
        margin: 0;
        max-width: 48px;
        max-height: 48px;
        text-align: center;
        white-space: nowrap;
        overflow-y: hidden;
        overflow-x: hidden;
      }
    </style>
  </head>
  <body>
    <table>
      <tr>""")

    print('<tr>')
    for i, char in enumerate(chars):
        print('<td>{}</td>'.format(char))
        if (i + 1) % 40 == 0:
            print('</tr><tr>')
    print('</tr>')
    print('</table></body></html>')


def main():
    chars = []
    with open('data/sampled_chars.txt') as f:
        for line in f:
            char = line.strip().split(' ')[1]
            # render_char(char)
            chars.append(char)
    create_html(chars)


if __name__ == '__main__':
    main()
