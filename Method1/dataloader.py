import os
import numpy as np
from PIL import Image

from nltk.tokenize import RegexpTokenizer

import torch
import torch.utils.data as data
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())


class CaltechBirds(data.Dataset):
    def __init__(self, img_root, caption_root,
                 classes_fllename, max_word_length,
                 vocab, img_transform=None):
        super(CaltechBirds, self).__init__()
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

        self.vocab = vocab
        self.max_word_length = max_word_length
        self.img_transform = img_transform

        if self.img_transform == None:
            self.img_transform = transforms.ToTensor()

        self.data = self._load_dataset(
            img_root, caption_root, classes_fllename)
        print("Load dataset size: ", len(self.data))

    def _load_dataset(self, img_root, caption_root, classes_filename):
        output = []

        with open(os.path.join(caption_root, classes_filename)) as f:
            lines = f.readlines()
            for line in lines:
                cls = line.replace('\n', '')
                filenames = os.listdir(os.path.join(caption_root, cls))
                for filename in filenames:
                    datum = load_lua(os.path.join(caption_root, cls, filename))
                    raw_desc = datum['char'].numpy()
                    desc, len_desc = self._get_word_vectors(raw_desc)
                    output.append({
                        'img': os.path.join(img_root, datum['img']),
                        'desc': desc,
                        'len_desc': len_desc
                    })
        return output

    def _get_word_vectors(self, desc):
        len_desc = []

        output = []

        for i in range(desc.shape[1]):
            sentence = self._nums2chars(desc[:, i])  # sentence
            tokens = split_sentence_into_words(sentence)  # word

            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))

            num_words = len(caption)
            x = np.zeros((self.max_word_length, 1), dtype='int64')
            x_len = num_words
            if num_words <= self.max_word_length:
                x[:num_words, 0] = caption
            else:
                ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:self.max_word_length]
                ix = np.sort(ix)
                x[:, 0] = [caption[idx] for idx in ix]
                x_len = self.max_word_length

            output.append(torch.Tensor(x))

            len_desc.append(x_len)

        return torch.stack(output), len_desc

    def _nums2chars(self, nums):
        chars = ''
        for num in nums:
            chars += self.alphabet[num - 1]
        return chars

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        file_name = datum['img']
        img = Image.open(file_name)
        img = self.img_transform(img)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        desc = datum['desc']
        len_desc = datum['len_desc']
        # randomly select one sentence
        selected = np.random.choice(desc.size(0))
        desc = desc[selected, ...]
        len_desc = len_desc[selected]
        return img, desc, len_desc, file_name


def get_loader(root, caption_root, vocab, split, img_transfrom,
               batch_size=1, shuffle=True, num_workers=4):
    classes_fllename = 'trainvalclasses.txt' \
        if split.lower() is 'train' else 'testclasses.txt'

    dataset = CaltechBirds(
        img_root=root,
        caption_root=caption_root,
        classes_fllename=classes_fllename,
        max_word_length=25,
        vocab=vocab,
        img_transform=img_transfrom)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader


if __name__ == '__main__':
    import pickle

    data_root = './Caltech200_birds/'
    img_root = os.path.join(data_root, 'CUB_200_2011/images')
    caption_root = os.path.join(data_root, 'cub_icml')
    vocab_file = os.path.join(caption_root, 'vocab.pkl')

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    print("Vocab size: ", len(vocab))

    transform = transforms.Compose([
        transforms.Resize(74),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    data_loader = get_loader(
        img_root, caption_root, vocab, split='train',
        img_transfrom=transform,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    for i, (images, targets, lengths, file_name) in enumerate(data_loader):
        print(images.shape)
        print(targets)
        print(lengths)
        break
