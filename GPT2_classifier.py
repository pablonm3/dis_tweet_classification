from torch_model_base import TorchModelBase
import torch
import torch.nn as nn
import torch.utils.data
from transformers import GPT2Tokenizer, GPT2Model
import random
from spacy.util import minibatch
import pandas as pd
from typing import List, Tuple
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")



def mean_across_all_tokens(hidden_states):
    return torch.mean(hidden_states[-1], dim=1)

def sum_all_tokens(hidden_states):
    return torch.sum(hidden_states[-1], dim=1)

def concat_all_tokens(hidden_states):
    batch_size, max_tokens, emb_dim = hidden_states[-1].shape
    return torch.reshape(hidden_states[-1], (batch_size, max_tokens * emb_dim))

class GPT2SequenceClassifierModel(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_classes: int,
            gpt_model_name: str,
            max_seq_length: int = 280,
            embedding_func=mean_across_all_tokens,
            combine_sentence_tokens=True,
            finetune_GPT2=False
    ):
        super(GPT2SequenceClassifierModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.model = GPT2Model.from_pretrained(
            gpt_model_name,
            output_hidden_states=True
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.combine_sentence_tokens = combine_sentence_tokens;
        self.embedding_func = embedding_func;
        self.max_length = max_seq_length
        self.finetune_GPT2 = finetune_GPT2
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _tokenize(self, text_list: List[str]) -> Tuple[torch.tensor, torch.tensor]:
        # Tokenize the text with the provided tokenizer
        input_ids = self.tokenizer.batch_encode_plus(text_list,
                                                     add_special_tokens=True,
                                                     max_length=self.max_length,
                                                     truncation=True,
                                                     pad_to_max_length=True
                                                     )["input_ids"]

        return torch.LongTensor(input_ids).to(device)

    def _tokenize_and_predict(self, text_list: List[str]) -> torch.tensor:
        input_ids_tensor = self._tokenize(text_list)
        out = self.model(input_ids=input_ids_tensor)
        hidden_states = out[2]
        if (self.combine_sentence_tokens):
            return self.embedding_func(hidden_states)
        else:
            return hidden_states[-1];


    def forward(self, text_list: List[str]):
        """
        :param input_ids: (torch.LongTensor of shape (batch_size, input_ids_length))
        :return: logits for class
        """
        if isinstance(text_list, pd.Series):
            text_list = text_list.tolist()
        if(self.finetune_GPT2):
            gpt_out = self._tokenize_and_predict(text_list)
        else:
            with torch.no_grad():
                # don't finetune GPT2 to save resources
                gpt_out = self._tokenize_and_predict(text_list)

        batch_size = len(text_list)
        assert gpt_out.shape == (batch_size, self.hidden_size)
        prediction_vector = self.fc1(gpt_out)  # (batch_size , max_len, num_classes)
        logits = torch.softmax(prediction_vector, dim=1)
        return logits


class GPT2Classifier(TorchModelBase):
    """GPT2 + NN head for classification problems.
    The network will work for any kind of classification task.

    Parameters
    ----------
    embed_dim: dimension of byte-pair/token embeddings generated by the model, check the model card(n_embd prop), since each model is compatible with only 1 no. of dimensions
    max_seq_length: max tokens in a sequence(n_positions param in hugging face model config), if sequenc is shorter will get padded
    checkpoint_path: file path to grab checkpoint from
    """
    def __init__(self,
            model_name="distilgpt2",
                 embed_dim=768,
                 max_seq_length=1024,
                 finetune_GPT2=True,
                 checkpoint_path=None,
                 base_dir='./',
                 classes=None,
                 **kwargs
                 ):
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.finetune_GPT2 = finetune_GPT2
        self.classes = classes
        self.checkpoint_path = checkpoint_path
        self.base_dir = base_dir
        super(GPT2Classifier, self).__init__(**kwargs)
        self.params += ['model_name']
        if self.classes:
            self.load_model()
        else:
            self.model = None # call fit() to set this

    def load_model(self):
        self.model = GPT2SequenceClassifierModel(
            hidden_size=self.embed_dim,
            num_classes=len(self.classes),
            gpt_model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            finetune_GPT2=self.finetune_GPT2
        )
        self.model.to(device)
        self.opt = self.optimizer(
            self.model.parameters()
        )
        if (self.checkpoint_path):
            # grab model and optimizer from checkpoint
            model_chk, opt_chk, self.current_ephoc = self.load_checkpoint()
            print("continuing training from checkpoint at ephoc: ", self.current_ephoc)
            self.model.load_state_dict(model_chk)
            self.opt.load_state_dict(opt_chk)
        else:
            self.current_ephoc = 0

    def fit(self, X, y):
        """Standard `fit` method.

        Parameters
        ----------
        X : np.array
        y : array-like
        Returns
        -------
        self

        """
        self.classes = list(set(y))
        if(not self.model):
            self.load_model()
        self.model.train()
        loss = nn.CrossEntropyLoss()
        print("Training... max iters: ", self.max_iter)
        for ephoc in (range(self.current_ephoc, self.max_iter)):
            print("ephoc no: ", ephoc)
            zipped_data = list(zip(X,y))
            random.shuffle(zipped_data)
            batches = minibatch(zipped_data, size=self.batch_size)
            for batch in batches:
                X_batch, y_batch = zip(*batch)
                batch_preds = self.model(X_batch)
                err = loss(batch_preds, torch.LongTensor(y_batch).to(device))
                # Backprop:
                self.opt.zero_grad()
                err.backward()
                self.opt.step()
            self.current_ephoc = ephoc + 1
            # save checkpoint
            checkpoint = {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(), "ephoc": self.current_ephoc}
            self.save_checkpoint(checkpoint)
        return self

    def save_checkpoint(self, checkpoint):
        path = self.base_dir + "/" + "GPT2_CLS_CHECKPOINT_ephoc" + str(checkpoint["ephoc"]) + ".pth.tar"
        print("saving checkpoint to file: ", path)
        torch.save(checkpoint, path)

    def load_checkpoint(self):
        print("loading checkpoint from: ", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path)
        return checkpoint["model"], checkpoint["optimizer"], checkpoint["ephoc"]

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
            preds = self.model(X)
            preds = preds.cpu().numpy()
            return preds

    def predict(self, X, batch_size=32):
        """Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Parameters
        ----------
        X : np.array
        batch_size: batch size to load on mem, dataset may be too big to fit it entirely on mem

        Returns
        -------
        list of length len(X)

        """
        batches = minibatch(X, size=batch_size)
        predictions = []
        for batch in batches:
            probs = self.predict_proba(batch)
            batch_predictions = [self.classes[i] for i in probs.argmax(axis=1)]
            predictions = predictions + batch_predictions
        return predictions