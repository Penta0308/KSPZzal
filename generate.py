import pickle

import modules
import numpy as np
import torch.utils.data
import os
from PIL import Image
import torch.optim
import torch.nn.functional as F
import shutil
import math

dataset = modules.ConcatDataset(
    modules.ImgWithoutTemplateIterableDataset(modules.config_get("dirs")[0], 0),
    modules.ImgWithoutTemplateIterableDataset(modules.config_get("dirs")[1], 1))

dataset_size = math.ceil(len(dataset))
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))  # train:valid = 0.9:0.1
np.random.seed(modules.SEED)
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=modules.BATCH_SIZE, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=modules.BATCH_SIZE, sampler=valid_sampler)

model = modules.Net()

print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
print(f'Sample: {dataset_size:,}')

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

itr = 1
epochs = 1
total_loss = 0
total_len = 0
total_correct = 0

for epoch in range(epochs):
    model.train()
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in train_loader:
        optimizer.zero_grad()

        output = model(batch[0])

        loss = criterion(output, batch[1])
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = pred.eq(batch[1])
        total_correct += correct.sum().item()
        total_len += len(batch[1])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"  Train Accuracy: {total_correct / total_len:.3f}, Loss: {total_loss / total_len:.4f}")
    total_loss = 0
    total_len = 0
    total_correct = 0

    model.eval()
    for batch in valid_loader:
        output = model(batch[0])

        loss = criterion(output, batch[1])
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = pred.eq(batch[1])
        total_correct += correct.sum().item()
        total_len += len(batch[1])

    print(f"  Test Accuracy: {total_correct / total_len:.3f}")
    total_len = 0
    total_correct = 0

    itr += 1

shutil.rmtree("data/model/", ignore_errors=True)
os.mkdir("data/model/")

with open('data/model/pytorch_model.bin', 'wb') as f:
    pickle.dump(model, f)