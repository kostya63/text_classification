import os
import pandas as pd
import numpy as np
import math
import time
import pathlib
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torchvision.transforms import ToTensor, Lambda, Compose
from prettytable import PrettyTable
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy

#------------------------------------------
batch_size = 8
learning_rate = 5
scheduler_step_size = 15
scheduler_gamma = 0.1
use_scheduling = True
test_batch_size = 1
epochs = 10
embedding_dim = 64

data_file = '/home/mkr/tc1/data/test_data.csv'
target_file = '/home/mkr/tc1/data/predictions.csv'
target_file_dialogs = '/home/mkr/tc1/data/dialogs.csv'

classes = [
    "Other",
    "Greeting",
    "Introduction",
    "Farewell"
]
correct_answers = [[0, 0], [0, 0], [0, 0], [0, 0]]

class TCDataset(Dataset):
	def __init__(self, data_file):
		self.source_data = pd.read_csv(data_file, sep = ';', header = 0)
		self.source_data.insert(loc=0, column='rowid', value=np.arange(len(self.source_data)))
		self.source_data.dropna(inplace = True)
		self.source_data = self.source_data.loc[self.source_data.iloc[:, 3]=='manager']
		#print(self.source_data)
	def __len__(self):
		return len(self.source_data)
	
	def __getitem__(self, idx):
		rowid = self.source_data.iloc[idx, 0]
		text_row = self.source_data.iloc[idx, 4]
		label = self.source_data.iloc[idx, 5]
		return label, text_row, rowid
     
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, rowid, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        #print('text:', text)
        #print('offsets:', offsets)
        predicted_label = model(text, offsets)
        #print('predicted:', predicted_label)
        #print('real label:', label)
        loss = loss_fn(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, rowid, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = loss_fn(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def yield_tokens(data_iter):
	for _, text, _ in data_iter:
		yield tokenizer(text)

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)
rowid_pipeline = lambda x: int(x)

def collate_batch(batch):
    label_list, text_list, rowid_list, offsets = [], [], [], [0]
    for (_label, _text, _rowid) in batch:
         label_list.append(label_pipeline(_label))
         rowid_list.append(rowid_pipeline(_rowid))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    rowid_list = torch.tensor(rowid_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), rowid_list.to(device), offsets.to(device)

#========== MAIN CODE ========

# Create data loaders.
#train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = True, drop_last = True)
#test_dataloader = DataLoader(training_data, batch_size=test_batch_size)
#eval_dataloader = DataLoader(test_data, batch_size=test_batch_size)

# Get cpu or gpu device for training
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("Using {} device".format(device))

tokenizer = get_tokenizer(None)
#tokenizer = spacy.load('ru_core_news_sm')


train_data = TCDataset(data_file)

train_len = train_data.__len__()
print('train_len:', train_len)

vocab = build_vocab_from_iterator(yield_tokens(train_data))
train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn = collate_batch)
test_dataloader = DataLoader(train_data, batch_size = 1, shuffle = False, drop_last = False, collate_fn = collate_batch)

#init the model
num_class = len(set([label for (label, text, rowid) in train_data]))
vocab_size = len(vocab)
model = TextClassificationModel(vocab_size, embedding_dim, num_class).to(device)
count_parameters(model)

### Pytorch-metric-learning setup ###
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = scheduler_gamma)

print('Lets get ready for the rumble with', epochs, end = ' ' )
print('epochs!')

total_accu = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(train_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

print("Done!")

torch.save(model.state_dict(), "model-100.pth")
print("Saved PyTorch Model State to model-100.pth")

#===== update target file =========
model.eval()
target_data = pd.read_csv(data_file, sep = ';', header = 0)
target_data.insert(loc=0, column='rowid', value=np.arange(len(target_data)))
target_data.insert(loc=6, column='predicted_class', value=None)
#print(target_data)
with torch.no_grad():
        for idx, (label, text, rowid, offsets) in enumerate(test_dataloader):
            predicted_label = model(text, offsets)
            target_idx = target_data.index[target_data['rowid'] == rowid[0].item()].tolist()[0]
            #print('target_idx: ', target_idx)
            target_data.at[target_idx, 'predicted_class'] = classes[predicted_label.argmax(1)]       

target_data.to_csv(target_file, index = False)
print('Saved!')

#====== dialogs analysis ==========
dialogs = pd.DataFrame(columns = ['dialog', 'status'])
#target_data.sort_values)

for i in range(target_data['dig_id'].max()):
	 dialog = target_data.loc[(target_data.iloc[:, 1 ] == i) & (target_data.iloc[:, 3] == 'manager')]
	 if target_data.iat[target_data['line_n'].idxmax(), 6] == 'Farewell':
		 dialogs.loc[len(dialogs.index)] = [i, 'Farewell found']
	 else:
		 dialogs.loc[len(dialogs.index)] = [i, 'Farewell NOT found']

dialogs.to_csv(target_file_dialogs, index = False)	 
	 
	 
	

