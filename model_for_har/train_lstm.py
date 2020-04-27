import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import CSIDataset
from models import LSTMClassifier
from schedulers import cosine, CyclicLR

from tqdm import tqdm

train_dataset = CSIDataset([
    "./dataset/bedroom_lviv/1",
    # "./dataset/bedroom_lviv/2",
    # "./dataset/bedroom_lviv/3",
    # "./dataset/vitalnia_lviv/1/",
    # "./dataset/vitalnia_lviv/2/",
    # "./dataset/vitalnia_lviv/3/",
    # "./dataset/vitalnia_lviv/4/"
])

val_dataset = CSIDataset([
    "./dataset/bedroom_lviv/4",
    # "./dataset/vitalnia_lviv/5/"
])

BATCH_SIZE = 64

trn_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

input_dim = 912
hidden_dim = 256
layer_dim = 3
output_dim = 7
seq_dim = 1

lr = 0.0005
n_epochs = 10
iterations_per_epoch = len(trn_dl)
best_acc = 0
patience, trials = 100, 0

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
model = model.cuda().double()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))

# training loop
print('Start model training')

for epoch in range(1, n_epochs + 1):

    model.train()
    for i, (x_batch, y_batch) in tqdm(enumerate(trn_dl), total=len(trn_dl), desc="Training epoch: "):

        if x_batch.shape[0] != BATCH_SIZE:  # TODO: fix this by padding sequences
            continue

        optimizer.zero_grad()

        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        out = model(x_batch)
        loss = criterion(out, y_batch)

        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    correct, total = 0, 0
    for x_val, y_val in tqdm(val_dl, total=len(val_dl), desc="Validation epoch: "):
        x_val, y_val = [t.cuda() for t in (x_val, y_val)]
        if x_val.shape[0] != BATCH_SIZE:  # TODO: fix this by padding sequences
            continue

        out = model(x_val)

        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        print("preds: ", preds)
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()

    acc = correct / total

    # if epoch % 5 == 0:
    print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), 'saved_models/best.pth')
        print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break
