from torch import nn
from constants import Const
from simple_lstm import SimpleLSTM
from crf import CRF

# or try the vectorized version:
# from crf_vectorized import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, nb_labels, emb_dim=5, hidden_dim=4):
        super().__init__()
        self.lstm = SimpleLSTM(
            vocab_size, nb_labels, emb_dim=emb_dim, hidden_dim=hidden_dim
        )
        self.crf = CRF(
            nb_labels,
            Const.BOS_TAG_ID,
            Const.EOS_TAG_ID,
            pad_tag_id=Const.PAD_TAG_ID,  # try setting pad_tag_id to None
            batch_first=True,
        )

    def forward(self, x, mask=None):
        emissions = self.lstm(x)
        score, path = self.crf.decode(emissions, mask=mask)
        return score, path

    def loss(self, x, y, mask=None):
        emissions = self.lstm(x)
        nll = self.crf(emissions, y, mask=mask)
        return nll
