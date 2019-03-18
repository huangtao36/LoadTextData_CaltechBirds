import os
import numpy as np
from PIL import Image

import nltk
from nltk.tokenize import RegexpTokenizer

import torch
import torch.utils.data as data
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms


def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())


def nums2chars(nums):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

    chars = ''
    for num in nums:
        chars += alphabet[num - 1]
    return chars


class CaltechBirds(data.Dataset):
    def __init__(self, img_root, caption_root, classes_fllename,
                 word_embedding, max_word_length, img_transform=None):
        super(CaltechBirds, self).__init__()

        self.max_word_length = max_word_length
        self.img_transform = img_transform

        if self.img_transform == None:
            self.img_transform = transforms.ToTensor()

        self.data = self._load_dataset(img_root, caption_root, classes_fllename, word_embedding)
        print("Load dataset size: ", len(self.data))

    def _load_dataset(self, img_root, caption_root, classes_filename, word_embedding):
        output = []

        with open(os.path.join(caption_root, classes_filename)) as f:
            lines = f.readlines()
            for line in lines:
                cls = line.replace('\n', '')
                filenames = os.listdir(os.path.join(caption_root, cls))
                for filename in filenames:
                    datum = load_lua(os.path.join(caption_root, cls, filename))
                    raw_desc = datum['char'].numpy()
                    desc, len_desc = self._get_word_vectors(raw_desc, word_embedding)
                    output.append({
                        'img': os.path.join(img_root, datum['img']),
                        'desc': desc,
                        'len_desc': len_desc
                    })
        return output

    def _get_word_vectors(self, desc, word_embedding):
        output = []
        len_desc = []
        for i in range(desc.shape[1]):
            words = nums2chars(desc[:, i])  # sentence
            words = split_sentence_into_words(words)  # word
            word_vecs = torch.Tensor([word_embedding[w] for w in words])  # vector [lens, 300]
            # zero padding to 50
            if len(words) < self.max_word_length:  # 50
                word_vecs = torch.cat((
                    word_vecs,
                    torch.zeros(self.max_word_length - len(words), word_vecs.size(1))
                ))
            output.append(word_vecs)
            len_desc.append(len(words))
        return torch.stack(output), len_desc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        img = Image.open(datum['img'])
        img = self.img_transform(img)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        desc = datum['desc']
        len_desc = datum['len_desc']
        # randomly select one sentence
        selected = np.random.choice(desc.size(0))
        desc = desc[selected, ...]
        len_desc = len_desc[selected]
        return img, desc, len_desc


if __name__ == '__main__':
    from pyfasttext import FastText

    dataroot = './Caltech200_birds'
    img_root = os.path.join(dataroot, 'CUB_200_2011/images')
    caption_root=os.path.join(dataroot, 'cub_icml')
    trainclasses_file = 'trainvalclasses.txt'
    fasttext_model = './PreTrainModel/wiki_en.bin'

    batch_size = 1

    word_embedding = FastText(fasttext_model)

    train_data = CaltechBirds(img_root,
                              caption_root,
                              trainclasses_file,
                              word_embedding,
                              max_word_length=50,
                              img_transform=transforms.Compose([
                                  transforms.Resize(74),
                                  transforms.RandomCrop(64),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()
                              ]))
    word_embedding = None

    train_loader = data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    for i, (img, desc, len_desc) in enumerate(train_loader):
        print('img: ', img.shape)
        print('desc: ', desc.shape)

        break

        # if i % 100 == 0:
        #     print(i)
