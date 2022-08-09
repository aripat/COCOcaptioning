import time

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import numpy as np
import sys


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-50 and replace top fully connected layer
        with a linear layer with output feature of dimenzion = embed_size."""
        super(EncoderCNN, self).__init__()

        # TODO: fonte per resnet
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*modules)
        # TODO: fonte per decoder e trasformazione lineare
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        print(features.size(0))

        # linearize tensors in batch
        features = features.view(features.size(0), -1)
        # linear transformation of CNN features in embed_size vector
        features = self.linear(features)
        features = self.bn(features)

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        # A simple lookup table that stores embeddings of a fixed dictionary and size:
        # used to store word embeddings and retrieve them using indices.
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # useful in order to make the prediction on the next token in the caption
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        captions = captions[:, :-1]
        embeddings = self.embed(captions)


        # Before the unsqueeze the encoder features have shape == (batch_size, embedding_size).
        # The first decoder input is the features from the encoder:
        # each encoder feature vector is concatenated to the corresponding embedded caption
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    # TODO
    def sample_beam_search(self):
        raise NotImplementedError("sample_beam_search not implemented")


## Functions to train and test the encoder - decoder architecture

def train(loader, encoder, decoder, loss_function, optimizer, vocab_size,
          epoch, num_train_batch):
    """Train the model for one epoch using the provided parameters"""

    # Switch to train mode
    encoder.train()
    decoder.train()

    # Keep track of train loss
    total_loss = 0.0

    # Start time for every 100 steps
    start_train_time = time.time()
    i_step = 0
    for batch in loader:
        # Obtain the batch
        if torch.cuda.is_available():
            images, captions = batch[0].cuda(), batch[1].cuda()
        else:
            images, captions = batch[0], batch[1]

        # Pass the inputs through the CNN-RNN model
        features = encoder(images)
        outputs = decoder(features, captions)

        # Calculate the batch loss
        loss = loss_function(outputs.view(-1, vocab_size), captions.view(-1))
        # Zero the gradients. Since the backward() function accumulates
        # gradients, and we don’t want to mix up gradients between minibatches,
        # we have to zero them out at the start of a new minibatch
        optimizer.zero_grad()
        # Backward pass to calculate the weight gradients
        loss.backward()
        # Update the parameters in the optimizer
        optimizer.step()

        total_loss += loss.item()

        # Get training statistics
        stats = "Epoch %d, Train step [%d/%d], %ds, Loss: %.4f" \
                % (epoch, i_step, num_train_batch, time.time() - start_train_time,
                   loss.item())

        # Print training statistics (on same line)
        print("\r" + stats, end="")
        sys.stdout.flush()

        i_step += 1

    return total_loss / num_train_batch