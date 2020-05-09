import torch
from torch.nn import functional as F
from tqdm import tqdm

# Cuda support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_train_metric(model, dl, criterion, BATCH_SIZE):
    model.eval()

    correct, total, total_loss = 0, 0, 0

    model.hidden = model.init_hidden(BATCH_SIZE)
    for x_val, y_val in tqdm(dl, total=len(dl), desc="Validation epoch: "):
        if x_val.size(0) != BATCH_SIZE:
            continue

        model.init_hidden(x_val.size(0))
        x_val, y_val = x_val.double().to(device), y_val.double().to(device)

        out = model(x_val)

        # out = out.view(out.size(0) * out.size(1), out.size(2))
        # y_val = y_val.view(y_val.size(0) * y_val.size(1))
        # print("y_val.size(0): ", y_val.size())

        loss = criterion(out, y_val.long())

        total_loss += loss.item()

        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()

    acc = correct / total

    return total_loss, correct, total, acc
