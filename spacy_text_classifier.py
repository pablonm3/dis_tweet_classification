#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import thinc.extra.datasets

import spacy
from spacy.util import minibatch, compounding



class SpacyClassifier():
    def __init__(self, n_iter):
        super(SpacyClassifier, self).__init__()
        self.nlp = spacy.blank("en")  # create blank Language class
        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "textcat" not in self.nlp.pipe_names:
            textcat = self.nlp.create_pipe(
                "textcat", config={"exclusive_classes": True, "architecture": "ensemble"}
            )
            self.nlp.add_pipe(textcat, last=True)
        self.textcat = self.nlp.get_pipe("textcat")
        self.n_iter = n_iter

    def fit(self, X, y):
        # add label to text classifier
        train_cats = [str(digit) for digit in y]
        labels = list(set(train_cats))
        for label in labels:
            self.textcat.add_label(label)
        train_cats_formatted = []
        for train_cat in train_cats:
            cats_dict = { i : False if i != train_cat else True for i in train_cats }
            train_cats_formatted.append(cats_dict)
        train_data = list(zip(X, [{"cats": cats} for cats in train_cats_formatted]))
        # get names of other pipes to disable them during training
        pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.nlp.begin_training()
            print("Training the model...")
            batch_sizes = compounding(4.0, 32.0, 1.001)
            for i in range(self.n_iter):
                print('iteration no. ', i)
                # batch up the examples using spaCy's minibatch
                random.shuffle(train_data)
                batches = minibatch(train_data, size=batch_sizes)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(texts, annotations, sgd=optimizer, drop=0.2)

    def predict(self, X):
        X_formatted = (self.nlp.tokenizer(text) for text in X)
        docs = list(self.textcat.pipe(X_formatted))
        cats = []
        for doc in docs:
            max_prob = 0;
            for key, value in doc.cats.items():
                if(value>max_prob):
                    max_prob = value
                    max_cat = key;
            cats.append(int(max_cat))
        return cats

