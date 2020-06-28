import math

import numpy as np
from operator import itemgetter
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

from torch_model_base import TorchModelBase
import torch.nn.functional as F

import sys

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"

def progress_bar(msg):
    """Simple over-writing progress bar."""
    sys.stderr.write('\r')
    sys.stderr.write(msg)
    sys.stderr.flush()


class TorchRNNDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, seq_lengths, y):
        assert len(sequences) == len(y)
        assert len(sequences) == len(seq_lengths)
        self.sequences = sequences
        self.seq_lengths = seq_lengths
        self.y = y

    @staticmethod
    def collate_fn(batch):
        X, seq_lengths, y = zip(*batch)
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        seq_lengths = torch.tensor(seq_lengths)
        y = torch.tensor(y)
        return X, seq_lengths, y

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (self.sequences[idx], self.seq_lengths[idx], self.y[idx])



class GlobalAttnManning2015(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""
    """This model is designed based on global attention presented in "Effective Approaches to Attention-based Neural Machine Translation" 
    and adapted by myself to work on text classification task, the original model was designed for seq2seq tasks
    """
    def __init__(self, hidden_size):
        super(GlobalAttnManning2015, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(self.hidden_size, 1) # for a vector(hidden state) return a score

    def forward(self, encoder_outputs):
        """Attend all encoder outputs conditioned on the previous hidden state of the decoder.

        After creating variables to store the attention energies, calculate their
        values for each encoder output and return the normalized values.

        Args:
            encoder_outputs: list of encoder outputs

        Returns:
             Normalized (0..1) energy values, dim: batch_size, hidden_size
        """
        energies = self._score(encoder_outputs)
        scores = F.softmax(energies, dim=1)
        #validate_softmax = torch.sum(scores, dim=1)
        scores = scores.squeeze()
        weighted_outputs = scores.view(scores.shape[0], scores.shape[1], 1) * encoder_outputs
        return torch.sum(weighted_outputs, dim=1) # weighted mean of encoder_outputs, divided by 1 ;)

    def _score(self, encoder_output):
        """Calculate the relevance of a particular encoder output"""
        energy = self.attention(encoder_output)
        return energy


class AttnShouPeng2016(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""
    """This model is designed based on attention presented in "Zhou, Peng, et al. "Attention-based bidirectional long short-term memory networks for relation classification." The 54th Annual Meeting of the Association for Computational Linguistics. 2016." 
    """
    def __init__(self, hidden_size):
        super(AttnShouPeng2016, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(self.hidden_size, 1) # for a vector(hidden state) return a score

    def forward(self, encoder_outputs):
        """Attend all encoder outputs conditioned on the previous hidden state of the decoder.

        After creating variables to store the attention energies, calculate their
        values for each encoder output and return the normalized values.

        Args:
            encoder_outputs: list of encoder outputs

        Returns:
             Normalized (0..1) energy values, dim: batch_size, hidden_size
        """
        energies = self._score(encoder_outputs)
        scores = F.softmax(energies, dim=1)
        #validate_softmax = torch.sum(scores, dim=1)
        scores = scores.squeeze()
        weighted_outputs = scores.view(scores.shape[0], scores.shape[1], 1) * encoder_outputs
        weighted_sum = torch.sum(weighted_outputs, dim=1)  # weighted mean of encoder_outputs, divided by 1 ;)
        return F.tanh(weighted_sum)

    def _score(self, encoder_output):
        """Calculate the relevance of a particular encoder output"""
        encoder_output = F.tanh(encoder_output)
        energy = self.attention(encoder_output)
        return energy



class TorchRNNClassifierModel(nn.Module):
    def __init__(self,
            vocab_size,
            embed_dim,
            embedding,
            use_embedding,
            hidden_dim,
            output_dim,
            bidirectional,
            attention,
            device):
        super(TorchRNNClassifierModel, self).__init__()
        self.use_embedding = use_embedding
        self.attention = attention
        self.device = device
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.hidden_size = hidden_dim
        # Graph
        if self.use_embedding:
            self.embedding = self._define_embedding(
                embedding, vocab_size, self.embed_dim)
            self.embed_dim = self.embedding.embedding_dim
        self.rnn = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional)
        if bidirectional:
            classifier_dim = hidden_dim * 2
        else:
            classifier_dim = hidden_dim
        if (self.attention == "GlobalAttnManning2015"):
            self.attention_layer = GlobalAttnManning2015(classifier_dim)
        if (self.attention == "AttnShouPeng2016"):
            self.attention_layer = AttnShouPeng2016(classifier_dim)
        self.classifier_layer = nn.Linear(classifier_dim, output_dim)

    def forward(self, X, seq_lengths):
        hidden_states = self.rnn_forward(X, seq_lengths, self.rnn)
        if(self.attention):
            hidden_states = self.attention_layer(hidden_states)
        logits = self.classifier_layer(hidden_states)
        return logits

    def rnn_forward(self, X, seq_lengths, rnn):
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        X = X.to(self.device, non_blocking=True)
        seq_lengths = seq_lengths.to(self.device)
        seq_lengths, sort_idx = seq_lengths.sort(0, descending=True)
        X = X[sort_idx]
        if self.use_embedding:
            embs = self.embedding(X)
        else:
            embs = X
        embs = torch.nn.utils.rnn.pack_padded_sequence(
           embs, batch_first=True, lengths=seq_lengths)
        all_states, last_states = rnn(embs)
        if(not self.attention):
            state = self.get_batch_final_states(last_states)
            if self.bidirectional:
                state = torch.cat((state[0], state[1]), dim=1)
        else:
            batch = len(sort_idx)
            seq_len = X.shape[1]
            if(self.bidirectional):
                num_directions = 2
            else:
                num_directions = 1
           # state = all_states.view(batch, seq_len, num_directions, self.hidden_size)
            # unpack sequence
            state, unpacked_len = \
                torch.nn.utils.rnn.pad_packed_sequence(
                    all_states, batch_first=True)
        _, unsort_idx = sort_idx.sort(0)
        state = state[unsort_idx]
        return state

    def get_batch_final_states(self, state):
        if self.rnn.__class__.__name__ == 'LSTM':
            return state[0].squeeze(0)
        else:
            return state.squeeze(0)

    @staticmethod
    def _define_embedding(embedding, vocab_size, embed_dim):
        if embedding is None:
            return nn.Embedding(vocab_size, embed_dim)
        else:
            embedding = torch.FloatTensor(embedding)
            return nn.Embedding.from_pretrained(embedding)


class TorchRNNClassifier(TorchModelBase):
    """LSTM-based Recurrent Neural Network for classification problems.
    The network will work for any kind of classification task.

    Parameters
    ----------
    vocab : list of str
        This should be the vocabulary. It needs to be aligned with
         `embedding` in the sense that the ith element of vocab
        should be represented by the ith row of `embedding`. Ignored
        if `use_embedding=False`.
    embedding : np.array or None
        Each row represents a word in `vocab`, as described above.
    use_embedding : bool
        If True, then incoming examples are presumed to be lists of
        elements of the vocabulary. If False, then they are presumed
        to be lists of vectors. In this case, the `embedding` and
        `embed_dim` arguments are ignored, since no embedding is needed
        and `embed_dim` is set by the nature of the incoming vectors.
    embed_dim : int
        Dimensionality for the initial embeddings. This is ignored
        if `embedding` is not None, as a specified value there
        determines this value. Also ignored if `use_embedding=False`.
    hidden_dim : int
        Dimensionality of the hidden layer.
    bidirectional : bool
        If True, then the final hidden states from passes in both
        directions are used.
    max_iter : int
        Maximum number of training epochs.
    eta : float
        Learning rate.
    optimizer : PyTorch optimizer
        Default is `torch.optim.Adam`.
    l2_strength : float
        L2 regularization strength. Default 0 is no regularization.
    device : 'cpu' or 'cuda'
        The default is to use 'cuda' iff available
    warm_start : bool
        If True, calling `fit` will resume training with previously
        defined trainable parameters. If False, calling `fit` will
        reinitialize all trainable parameters. Default: False.

    """
    def __init__(self,
            vocab,
            embedding=None,
            use_embedding=True,
            embed_dim=50,
            bidirectional=False,
            attention=None,
            **kwargs):
        self.vocab = vocab
        self.attention = attention
        self.embedding = embedding
        self.use_embedding = use_embedding
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        super(TorchRNNClassifier, self).__init__(**kwargs)
        self.params += ['embed_dim', 'embedding', 'use_embedding', 'attention', 'bidirectional']
        # The base class has this attribute, but this model doesn't,
        # so we remove it to avoid misleading people:
        delattr(self, 'hidden_activation')
        self.params.remove('hidden_activation')

    def build_dataset(self, X, y):
        X, seq_lengths = self._prepare_dataset(X)
        return TorchRNNDataset(X, seq_lengths, y)

    def build_graph(self):
        return TorchRNNClassifierModel(
            vocab_size=len(self.vocab),
            embedding=self.embedding,
            use_embedding=self.use_embedding,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_classes_,
            bidirectional=self.bidirectional,
            device=self.device,
            attention=self.attention
        )

    def fit(self, X, y, **kwargs):
        """Standard `fit` method.

        Parameters
        ----------
        X : np.array
        y : array-like
        kwargs : dict
            For passing other parameters. If 'X_dev' is included,
            then performance is monitored every 10 epochs; use
            `dev_iter` to control this number.

        Returns
        -------
        self

        """
        # Incremental performance:
        X_dev = kwargs.get('X_dev')
        if X_dev is not None:
            dev_iter = kwargs.get('dev_iter', 10)
        # Data prep:
        self.classes_ = sorted(set(y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        y = [class2index[label] for label in y]
        dataset = self.build_dataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn)
        if not self.use_embedding:
            # Infer `embed_dim` from `X` in this case:
            self.embed_dim = X[0][0].shape[0]
        # Graph:
        if not self.warm_start or not hasattr(self, "model"):
            self.model = self.build_graph()
            self.opt = self.optimizer(
                self.model.parameters(),
                lr=self.eta,
                weight_decay=self.l2_strength)
        self.model.to(self.device)
        self.model.train()
        # Make sure this value is up-to-date; self.`model` might change
        # it if it creates an embedding:
        self.embed_dim = self.model.embed_dim
        # Optimization:
        loss = nn.CrossEntropyLoss()
        # Train:
        for iteration in range(1, self.max_iter+1):
            epoch_error = 0.0
            for X_batch, batch_seq_lengths, y_batch in dataloader:
                y_batch = y_batch.to(self.device, non_blocking=True)
                batch_preds = self.model(X_batch, batch_seq_lengths)
                err = loss(batch_preds, y_batch)
                epoch_error += err.item()
                # Backprop:
                self.opt.zero_grad()
                err.backward()
                self.opt.step()
            # Incremental predictions where possible:
            if X_dev is not None and iteration > 0 and iteration % dev_iter == 0:
                self.dev_predictions[iteration] = self.predict(X_dev)
                self.model.train()
            self.errors.append(epoch_error)
            progress_bar("Finished epoch {} of {}; error is {}".format(
                iteration, self.max_iter, epoch_error))
        return self

    def predict_proba(self, X):
        """Predicted probabilities for the examples in `X`.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        np.array with shape (len(X), self.n_classes_)

        """
        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            X, seq_lengths = self._prepare_dataset(X)
            preds = self.model(X, seq_lengths)
            preds = torch.softmax(preds, dim=1).cpu().numpy()
            return preds

    def predict(self, X):
        """Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        list of length len(X)

        """
        probs = self.predict_proba(X)
        return [self.classes_[i] for i in probs.argmax(axis=1)]

    def _prepare_dataset(self, X):
        """Internal method for preprocessing a set of examples. If
        `self.use_embedding=True`, then `X` is transformed into a list
        of lists of indices. Otherwise, `X` is assumed to already
        contain the vectors we want to process. In both situations,
        we measure the lengths of the sequences in `X`.

        Parameters
        ----------
        X : list of lists of tokens, or list of np.array of vectors

        Returns
        -------
        list of lists of ints, or list of np.array of vectors,
        and `torch.LongTensor` of sequence lengths.

        """
        new_X = []
        seq_lengths = []
        if self.use_embedding:
            index = dict(zip(self.vocab, range(len(self.vocab))))
            unk_index = index['$UNK']
            for ex in X:
                seq = [index.get(w, unk_index) for w in ex]
                seq = torch.tensor(seq)
                new_X.append(seq)
                seq_lengths.append(len(seq))
        else:
            new_X = [torch.tensor(ex) for ex in X]
            seq_lengths = [len(ex) for ex in X]
        return new_X, torch.tensor(seq_lengths)


def simple_example(initial_embedding=False, use_embedding=True):
    vocab = ['a', 'b', '$UNK']

    # No b before an a
    train = [
        [list('ab'), 'good'],
        [list('aab'), 'good'],
        [list('abb'), 'good'],
        [list('aabb'), 'good'],
        [list('ba'), 'bad'],
        [list('baa'), 'bad'],
        [list('bba'), 'bad'],
        [list('bbaa'), 'bad'],
        [list('aba'), 'bad']
    ]

    test = [
        [list('baaa'), 'bad'],
        [list('abaa'), 'bad'],
        [list('bbaa'), 'bad'],
        [list('aaab'), 'good'],
        [list('aaabb'), 'good']
    ]

    if initial_embedding:
        import numpy as np
        # `embed_dim=60` to make sure that it gets changed internally:
        embedding = np.random.uniform(
            low=-1.0, high=1.0, size=(len(vocab), 60))
    else:
        embedding = None

    mod = TorchRNNClassifier(
        vocab=vocab,
        max_iter=100,
        embed_dim=50,
        embedding=embedding,
        use_embedding=use_embedding,
        bidirectional=False,
        hidden_dim=50)

    X, y = zip(*train)
    X_test, y_test = zip(*test)

    # Just to illustrate how we can process incoming sequences of
    # vectors, we create an embedding and use it to preprocess the
    # train and test sets:
    if not use_embedding:
        import numpy as np
        from copy import copy
        # `embed_dim=60` to make sure that it gets changed internally:
        embedding = np.random.uniform(
            low=-1.0, high=1.0, size=(len(vocab), 60))
        X = [[embedding[vocab.index(w)] for w in ex] for ex in X]
        # So we can display the examples sensibly:
        X_test_orig = copy(X_test)
        X_test = [[embedding[vocab.index(w)] for w in ex] for ex in X_test]
    else:
        X_test_orig = X_test

    mod.fit(X, y)

    preds = mod.predict(X_test)

    print("\nPredictions:")

    for ex, pred, gold in zip(X_test_orig, preds, y_test):
        score = "correct" if pred == gold else "incorrect"
        print("{0:>6} - predicted: {1:>4}; actual: {2:>4} - {3}".format(
            "".join(ex), pred, gold, score))


if __name__ == '__main__':
    simple_example()
