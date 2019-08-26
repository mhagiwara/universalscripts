"""Render a table of all chars in an HTML file."""

IMG_WIDTH = 32
IMG_HEIGHT = 32

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
            chars.append(char)
    create_html(chars)


if __name__ == '__main__':
    main()
