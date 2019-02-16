import torch
from torch import nn


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model copied from:
    https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    """

    def __init__(self, vocab_size, nb_labels, emb_dim=10, hidden_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, nb_labels)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2),
            torch.randn(2, batch_size, self.hidden_dim // 2),
        )

    def forward(self, batch_of_sentences):
        self.hidden = self.init_hidden(batch_of_sentences.shape[0])
        x = self.emb(batch_of_sentences)
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.hidden2tag(x)
        return x
