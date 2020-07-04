import unittest
import pandas as pd
from sklearn.metrics import classification_report

from torch_rnn_classifier_attn import TorchRNNClassifier
from spacy_text_classifier import SpacyClassifier
from GPT2_classifier import GPT2Classifier
from spacy_example import main
import pandas as pd
import numpy as np
import torch

train_df = pd.read_csv("./data/train_test.csv")
y_train = train_df['target'].tolist()
X_train_text = train_df['text'].tolist()
X_train = torch.load('X_train_vectors.pt')


@unittest.skip("Skipping RNN cls")
class RNNClassifierTestCase(unittest.TestCase):
    def test_finish_ok(self):
        torch_rnn = TorchRNNClassifier(
            vocab=[],
            use_embedding=False,
            attention="AttnShouPeng2016",
            bidirectional=True,
            hidden_dim=50,
            max_iter=2,
            eta=0.05)
        breakpoint()
        _ = torch_rnn.fit(X_train, y_train)
        breakpoint()
        predictions = torch_rnn.predict(X_train)
        breakpoint()
        print(classification_report(y_train, predictions))

        self.assertEqual(True, True)

@unittest.skip("Skipping RNN cls")
class SpacylassifierTestCase(unittest.TestCase):
    def test_finish_ok(self):
        classifier = SpacyClassifier(n_iter=2)
        classifier.fit(X_train_text, y_train)
        breakpoint()
        predictions = classifier.predict(X_train_text)
        breakpoint()
        print(classification_report(y_train, predictions))

        self.assertEqual(True, True)

@unittest.skip("Skipping RNN cls")
class SpacylassifierBertTestCase(unittest.TestCase):
    def test_finish_ok(self):
        classifier = SpacyClassifierBert(n_iter=2)
        classifier.fit(X_train_text, y_train)
        breakpoint()
        predictions = classifier.predict(X_train_text)
        breakpoint()
        print(classification_report(y_train, predictions))
        self.assertEqual(True, True)

#@unittest.skip("Skipping RNN cls")
class GPT2ClassifierTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        train_df = pd.read_csv("./data/train_test.csv")
        self.y_train = train_df['target'].tolist()[0:100]
        self.X_train_text = train_df['text'].tolist()[0:100]

    def test_finish_ok(self):
        classifier = GPT2Classifier(max_iter=1, batch_size=32)
        classifier.fit(self.X_train_text, self.y_train)
        breakpoint()
        predictions = classifier.predict(self.X_train_text)
        breakpoint()
        print(classification_report(self.y_train, predictions))
        self.assertEqual(True, True)




if __name__ == '__main__':
    unittest.main()
