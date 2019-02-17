"""
Code adapted from:
https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
"""

import torch
import torch.optim as optim

from constants import Const
from bilstm_crf import BiLSTM_CRF

# for reproducibility
torch.manual_seed(1)


def get_word_and_tag_vocab(training_data):
    word_to_ix = {
        Const.UNK_TOKEN: Const.UNK_ID,
        Const.PAD_TOKEN: Const.PAD_ID,
        Const.BOS_TOKEN: Const.BOS_ID,
        Const.EOS_TOKEN: Const.EOS_ID,
    }

    tag_to_ix = {
        Const.PAD_TAG_TOKEN: Const.PAD_TAG_ID,
        Const.BOS_TAG_TOKEN: Const.BOS_TAG_ID,
        Const.EOS_TAG_TOKEN: Const.EOS_TAG_ID,
    }

    for sentence, tags in training_data:
        for word, tag in zip(sentence, tags):
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, tag_to_ix


def prepare_sequence(seq, stoi):
    return torch.tensor([stoi[w] for w in seq], dtype=torch.long)


def ids_to_tags(seq, itos):
    return [itos[x] for x in seq]


if __name__ == "__main__":

    training_data = [
        (
            "the wall street journal reported today that apple corporation made money".split(),  # NOQA
            "B I I I O O O B I O O".split(),
        ),
        ("georgia tech is a university in georgia".split(), "B I O O O O B".split()),
    ]

    max_sent_size = max([len(tags) for words, tags in training_data])
    print("Max sentence size:", max_sent_size)

    word_to_ix, tag_to_ix = get_word_and_tag_vocab(training_data)
    print("Tag vocab:", tag_to_ix)

    print('Dataset tags:')
    for words, tags in training_data:
        print(' '.join(tags))

    # prepare sequence for each sample in our training data
    x_sent_1 = prepare_sequence(training_data[0][0], word_to_ix)
    x_tags_1 = prepare_sequence(training_data[0][1], tag_to_ix)
    x_sent_2 = prepare_sequence(training_data[1][0], word_to_ix)
    x_tags_2 = prepare_sequence(training_data[1][1], tag_to_ix)

    # create a batch of 2 samples with their proper padding
    x_sent = torch.full((2, max_sent_size), Const.PAD_ID, dtype=torch.long)
    x_sent[0, : x_sent_1.shape[0]] = x_sent_1
    x_sent[1, : x_sent_2.shape[0]] = x_sent_2

    # create a batch of 2 samples with their proper padding
    x_tags = torch.full((2, max_sent_size), Const.PAD_TAG_ID, dtype=torch.long)
    x_tags[0, : x_tags_1.shape[0]] = x_tags_1
    x_tags[1, : x_tags_2.shape[0]] = x_tags_2

    # mask tensor with shape (batch_size, max_sent_size)
    mask = (x_tags != Const.PAD_TAG_ID).float()

    # get a reversed dict mapping int to str
    ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}

    # see bilstm_crf.py
    model = BiLSTM_CRF(len(word_to_ix), len(tag_to_ix))
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    print('Predictions before training:')
    with torch.no_grad():
        scores, seqs = model(x_sent, mask=mask)
        for score, seq in zip(scores, seqs):
            str_seq = " ".join(ids_to_tags(seq, ix_to_tag))
            print('%.2f: %s' % (score.item(), str_seq))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(300):  # normally you would NOT do 300 epochs, it is toy data
        
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = x_sent
        targets = x_tags

        # Step 3. Run our forward pass.
        loss = model.loss(sentence_in, targets, mask=mask)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

    # Check predictions after training
    print('Predictions after training:')
    with torch.no_grad():
        scores, seqs = model(x_sent, mask=mask)
        for score, seq in zip(scores, seqs):
            str_seq = " ".join(ids_to_tags(seq, ix_to_tag))
            print('%.2f: %s' % (score.item(), str_seq))
