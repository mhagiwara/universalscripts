"""A Python script to sample random characters from unicode code blocks."""

import numpy as np
import glob
from collections import defaultdict


CHARS_PER_FILE = 100


def gather_ud_files(ud_path):
    files = []
    for file_path in glob.glob('{}/*/*.txt'.format(ud_path)):
        file_name = file_path.split('/')[-1]
        if file_name.startswith('LICENSE.txt') or file_name.startswith('README.txt'):
            continue
        files.append((file_path, file_name))

    # group by language
    langs = defaultdict(list)
    for file in files:
        file_path, file_name = file
        lang = file_name.split('_')[0]
        langs[lang].append((file_path, file_name))

    return langs


def main():
    np.random.seed(314)

    all_chars = set()
    langs = gather_ud_files('data/ud-treebanks-v2.3')
    for lang, files in langs.items():
        lang_content = ''
        for file in files:
            file_path, file_name = file
            with open(file_path) as f:
                content = f.read()

                content = content.replace(' ', '').replace('_', '').replace('\n', '')
                if not content:
                    continue

                lang_content += content

        if not lang_content:
            continue
        chosen_chars = np.random.choice(list(lang_content), CHARS_PER_FILE)
        print(lang, list(chosen_chars))
        all_chars.update(chosen_chars)

    with open('data/sampled_chars.txt', mode='w') as f:
        for char in sorted(all_chars):
            f.write('u{:04x} {}\n'.format(ord(char), char))


if __name__ == '__main__':
    main()
