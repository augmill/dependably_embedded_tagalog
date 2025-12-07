from Classifier import *
from datasets import load_dataset
from sklearn.model_selection import train_test_split

batch_size = 64

seed = 42
torch.manual_seed(seed)
# torch.set_default_dtype(torch.float64) 

fake_news_data = load_dataset("jcblaise/fake_news_filipino")["train"]
data = [(item['article'], item['label']) for item in fake_news_data]
X_train, X_test, y_train, y_test = train_test_split(
    [article[0] for article in data],
    [label[1] for label in data], 
    train_size=0.8, 
    test_size=0.2, 
    random_state=seed, 
    shuffle=True
)

train_data = [X_train, y_train]
split = int(len(X_test)*0.5)
val_data = [X_test[:split], y_test[:split]]
test_data = [X_test[split+1:], y_test[split+1:]]





bert_model = "paulbontempo/bert-tagalog-dependency-cl" 
model_type = "cl"

train_batches = make_batches(train_data, batch_size, bert_model)
dev_batches = make_batches(val_data, batch_size, bert_model)
test_batches = make_batches(test_data, batch_size, bert_model)

torch.save(train_batches, f"../data/{model_type}_train.pt")
torch.save(dev_batches, f"../data/{model_type}_dev.pt")
torch.save(test_batches, f"../data/{model_type}_test.pt")

cl_train_batches = torch.load(f"../data/cl_train.pt")
cl_dev_batches = torch.load(f"../data/cl_dev.pt")
cl_test_batches = torch.load(f"../data/cl_test.pt")

cl_model = Classifier(
    dim_in=768,
    drop = 0.5
)

train_losses = fit(
    model=cl_model,
    train_data=cl_train_batches,
    dev_data=cl_dev_batches,
    opt=torch.optim.AdamW(params=cl_model.parameters(), lr=1e-3),
    loss_fn=nn.BCEWithLogitsLoss(),
    epochs=20
)

bert_model = "paulbontempo/bert-tagalog-mlm-stage1"  
model_type = "s1" 

train_batches = make_batches(train_data, batch_size, bert_model)
dev_batches = make_batches(val_data, batch_size, bert_model)
test_batches = make_batches(test_data, batch_size, bert_model)

torch.save(train_batches, f"../data/{model_type}_train.pt")
torch.save(dev_batches, f"../data/{model_type}_dev.pt")
torch.save(test_batches, f"../data/{model_type}_test.pt")

s1_train_batches = torch.load(f"../data/s1_train.pt")
s1_dev_batches = torch.load(f"../data/s1_dev.pt")
s1_test_batches = torch.load(f"../data/s1_test.pt")

s1_model = Classifier(
    dim_in=768,
    drop=0.5
)

train_losses = fit(
    model=s1_model,
    train_data=s1_train_batches,
    dev_data=s1_dev_batches,
    opt=torch.optim.AdamW(params=s1_model.parameters(), lr=1e-3),
    loss_fn=nn.BCEWithLogitsLoss(),
    epochs=10
)

bert_model = "google-bert/bert-base-multilingual-cased"
model_type = "base"

train_batches = make_batches(train_data, batch_size, bert_model)
dev_batches = make_batches(val_data, batch_size, bert_model)
test_batches = make_batches(test_data, batch_size, bert_model)

torch.save(train_batches, f"../data/{model_type}_train.pt")
torch.save(dev_batches, f"../data/{model_type}_dev.pt")
torch.save(test_batches, f"../data/{model_type}_test.pt")

base_train_batches = torch.load(f"../data/base_train.pt")
base_dev_batches = torch.load(f"../data/base_dev.pt")
base_test_batches = torch.load(f"../data/base_test.pt")

base_model = Classifier(
    dim_in=768,
    drop=0.5
)

train_losses = fit(
    model=base_model,
    train_data=base_train_batches,
    dev_data=base_dev_batches,
    opt=torch.optim.AdamW(params=base_model.parameters(), lr=1e-4),
    loss_fn=nn.BCEWithLogitsLoss(),
    epochs=10
)

