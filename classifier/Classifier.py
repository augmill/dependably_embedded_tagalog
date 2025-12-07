import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from transformers import AutoModel, AutoTokenizer

seed=42

torch.manual_seed(seed)

# Load model and tokenizer


"""
do all at once before to make data loader 
"""

def init_weights(model):
    # NOTE: add instances for other types
    if isinstance(model, nn.Linear):
        torch.nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
        model.bias.data.fill_(0.01) 

class Classifier(nn.Module):
    def __init__(
            self, 
            dim_in: int, 
            drop: float,
            *args, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dim_in = dim_in
        self.dim_hidden = int((dim_in * 2/3) + 1) # 1 for binary output
        self.layer1 = nn.Linear(self.dim_in, self.dim_hidden)
        self.hidden = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.layer2 = nn.Linear(self.dim_hidden, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.sigmoid = nn.Sigmoid()
        self.model = nn.Sequential(
            self.layer1, 
            self.relu,
            self.drop,
            self.hidden, 
            self.relu,
            self.drop,
            self.layer2
        )
        self.model.apply(init_weights)

    def forward(self, input: torch.Tensor):
        return self.model(input)

def training(
        model: nn.Module,
        data, 
        opt: torch.optim, 
        loss_fn
):
    losses = []
    model.train()
    for batch in tqdm(data):
        opt.zero_grad()
        logits = model(batch[0])
        # print(batch[1].shape)
        loss = loss_fn(logits, batch[1].reshape(len(logits), 1).float())
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
    return losses

def validate(
        model: nn.Module,
        data, 
        # metric_fn
        acc, 
        f1
):
    model.eval()
    # losses = []
    accs = []
    f1s = []
    with torch.no_grad():
        for batch in tqdm(data):
            logits = model(batch[0])
            # loss = metric_fn(logits, batch[1])
            # losses.append(loss.detach().item())
            # accs.append(acc(torch.argmax(logits, dim=1), batch[1]).item())
            # f1s.append(f1(torch.argmax(logits, dim=1), batch[1]).item())
            accs.append(acc(logits, batch[1].reshape(len(logits), 1)))
            f1s.append(f1(logits, batch[1].reshape(len(logits), 1)))

    return accs, f1s
    # return losses

def make_batches(data, batch_size, bert_model):
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    bert = AutoModel.from_pretrained(bert_model)
    for param in bert.parameters():
        param.requires_grad = False
    articles = data[0]
    labels = data[1]
    batches = []
    for i in range(0, len(articles), batch_size):
        batches.append([bert(**tokenizer(articles[i:i+batch_size], return_tensors="pt", padding=True, truncation=True)).pooler_output, 
                        torch.tensor(labels[i:i+batch_size])])
    return batches

def fit(
        # self,
        model: nn.Module, 
        train_data: tuple, 
        dev_data: tuple,
        opt: torch.optim, 
        loss_fn,
        epochs: int
):
    # model.eval() #NOTE: may not be useful 
    # with torch.no_grad():
    train_losses = []
    dev_losses = []

    acc = Accuracy(task="binary")
    f1 = F1Score(task="binary")
    # train_batches = make_batches(train_data, batch_size)
    # dev_batches = make_batches(dev_data, batch_size)
    for epoch in range(epochs):
        print("-"*25 + f"epoch {epoch+1}" + "-"*25)
        # print(f"mdoel: {model}\ndata: {train_data}\nopt {opt}\nloss: {loss_fn}")
        train_loss = training(model, train_data, opt, loss_fn)
        train_losses.append(train_loss)
        print(f"Training loss: {sum(train_loss)/len(train_loss)}")
        # print(f"Training loss: {train_loss}")
        accs, f1s = validate(model, dev_data, acc, f1)
        # dev_losses.append(dev_loss)
        print(f"Validation accuracy: {sum(accs)/len(accs)}")
        print(f"Validation f1: {sum(f1s)/len(f1s)}")
        # print(f"Validation loss: {val_loss}")

    return train_losses #, dev_losses
