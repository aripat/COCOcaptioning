import os
import pandas as pd
from spacy.lang.en import English
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
# TODO check alive_bar
from alive_progress import alive_bar
import string

punctuations = set(string.punctuation)

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        spacy_eng = English()
        tokenizer_eng = spacy_eng.tokenizer
        return [token.text.lower() for token in tokenizer_eng(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        with alive_bar(len(sentence_list)) as bar:
            for sentence in sentence_list:
                for word in self.tokenizer_eng(sentence):
                    if word not in punctuations:
                        if word not in frequencies:
                            frequencies[word] = 1

                        else:
                            frequencies[word] += 1

                        if frequencies[word] == self.freq_threshold:
                            self.itos[idx] = word
                            self.stoi[word] = idx
                            idx += 1
                bar()

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

    def ids_to_words(self, indexes):
        return [self.itos[idx] for idx in indexes]



class COCODataset(Dataset):
    def __init__(self, img_dir, captions_file, transform=None, freq_threshold=5):
        self.transform = transform
        self.img_dir = img_dir
        self.df = pd.read_csv(captions_file)

        self.captions = self.df["caption"]
        self.files = self.df["file"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.files[index]
        caption = self.captions[index]

        img = Image.open(os.path.join(self.img_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]  # add dimension for the batch
        imgs = torch.cat(imgs, dim=0)  # tensor to collect all imgs
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets


def get_loader(
        img_dir,
        captions_file,
        transform=None,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True
):
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),  # smaller edge of image resized to 256
                transforms.CenterCrop(224),  # get 224x224 crop from center (224 is the ResNet input size
                transforms.ToTensor(),
            ]
        )

    dataset = COCODataset(img_dir=img_dir, captions_file=captions_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader


def main():
    img_dir = './data/cocoapi/images/train2017'
    captions_file = './data/anns-50.csv'

    dataloader = get_loader(img_dir=img_dir, captions_file=captions_file)

    for idx, (img, caption) in enumerate(dataloader):
        print(f'image shape :: ' + str(img.shape))
        print(f'caption shape :: ' + str(caption.shape))
        print('\n')


if __name__ == '__main__':
    main()
