import os
import pickle
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from torch.utils.serialization import load_lua


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


def _nums2chars(nums):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    chars = ''
    for num in nums:
        chars += alphabet[num - 1]
    return chars


def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())


def build_vocab_for_birds(caption_root, vocab_file):
    classes_file = ['trainvalclasses.txt', 'testclasses.txt']

    counter = Counter()

    for class_ in classes_file:
        with open(os.path.join(caption_root, class_)) as f:
            lines = f.readlines()
            for line in lines:
                cls = line.replace('\n', '')
                filenames = os.listdir(os.path.join(caption_root, cls))
                for filename in filenames:
                    datum = load_lua(os.path.join(caption_root, cls, filename))
                    raw_desc = datum['char'].numpy()
                    for i in range(raw_desc.shape[1]):
                        sentence = _nums2chars(raw_desc[:, i])  # sentence
                        tokens = split_sentence_into_words(sentence)  # word
                        counter.update(tokens)

    # 过滤掉一些拼错的词
    threshold = 2
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    len(words)

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for _, word in enumerate(words):
        vocab.add_word(word)

    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)

    print("Save the Vocab in to: ", vocab_file)


if __name__ == '__main__':

    caption_root = '/home/OpenResource/Datasets/Caltech200_birds/cub_icml'
    vocab_file = os.path.join(caption_root, 'vocab.pkl')
    build_vocab_for_birds(caption_root, vocab_file)
