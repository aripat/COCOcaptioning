import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import sys
import math
import torch.utils.data as data
import numpy as np
import os
import requests
import time

from dataloader import get_loader
from model import EncoderCNN, DecoderRNN, train


def main():
    # Set values for the training variables
    batch_size = 32  # batch size
    embed_size = 256  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder
    num_epochs = 1  # number of training epochs
    lr = 0.001

    # Build data loader, applying the transforms
    img_dir = './data/cocoapi/images/train2017'
    captions_file = './data/anns-50.csv'
    loader = get_loader(img_dir=img_dir, captions_file=captions_file, batch_size=batch_size)
    # TODO: insert right files for validation
    validation_loader = get_loader(img_dir=img_dir, captions_file=captions_file, batch_size=batch_size)

    # Initialize the encoder and decoder
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size=len(loader.dataset.vocab))

    # Move models to GPU if CUDA is available
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Define the loss function lf
    if torch.cuda.is_available():
        lf = nn.CrossEntropyLoss().cuda()
    else:
        lf = nn.CrossEntropyLoss()

    # Specify the learnable parameters
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())

    # Define the optimizer
    optimizer = torch.optim.Adam(params=params, lr=lr)

    # Set the total number of training and validation steps per epoch
    num_train_batch = math.ceil(len(loader.dataset) / batch_size)
    num_val_batch = math.ceil(len(loader.dataset) / batch_size)
    print("Number of train batches:", num_train_batch)
    print("Number of validation steps:", num_val_batch)

    losses = []
    for epoch in range(0, num_epochs):
        train_loss = train(loader, encoder, decoder, lf, optimizer,
                           vocab_size=len(loader.dataset.vocab),
                           epoch=epoch,
                           num_train_batch=num_train_batch)

        losses.append(train_loss)

    print("Losses")
    print(losses)


if __name__ == "__main__":
    main()