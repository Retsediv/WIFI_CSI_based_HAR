import logging

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import CSIDataset
from metrics import get_train_metric
from models import LSTMClassifier, FCNBaseline, SimpleLSTMClassifier, InceptionModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

# Cuda support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

logging.info("Device: {}".format(device))

# LSTM Model parameters
input_dim = 468  # 114 subcarriers * 4 antenna_pairs * 2 (amplitude + phase)
hidden_dim = 256
layer_dim = 2
output_dim = 7
dropout_rate = 0.0
bidirectional = False
SEQ_DIM = 1024
DATA_STEP = 8

BATCH_SIZE = 4
EPOCHS_NUM = 100
LEARNING_RATE = 0.00146

class_weights = torch.Tensor([0.113, 0.439, 0.0379, 0.1515, 0.0379, 0.1212, 0.1363]).double().to(device)
class_weights_inv = 1 / class_weights
logging.info("class_weights_inv: {}".format(class_weights_inv))


def load_data():
    logging.info("Loading data...")

    train_dataset = CSIDataset([
        "./dataset/bedroom_lviv/1",
        # "./dataset/bedroom_lviv/2",
        # "./dataset/bedroom_lviv/3",
        # "./dataset/vitalnia_lviv/1/",
        # "./dataset/vitalnia_lviv/2/",
        # "./dataset/vitalnia_lviv/3/",
        # "./dataset/vitalnia_lviv/4/"
    ], SEQ_DIM, DATA_STEP)

    val_dataset = train_dataset
    # val_dataset = CSIDataset([
    #     "./dataset/bedroom_lviv/4",
    #     # "./dataset/vitalnia_lviv/5/"
    # ], SEQ_DIM)

    logging.info("Data is loaded...")

    trn_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return trn_dl, val_dl


def train():
    patience, trials, best_acc = 100, 0, 0
    trn_dl, val_dl = load_data()

    # model = FCNBaseline(SEQ_DIM, output_dim)
    # model = LSTMClassifier(input_dim, hidden_dim, layer_dim, dropout_rate, bidirectional, output_dim, BATCH_SIZE)
    # model = SimpleLSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim, dropout_rate, bidirectional,)
    model = InceptionModel(3, SEQ_DIM, input_dim, 12, 15, True, output_dim)
    model = model.double().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_inv)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    # training loop
    logging.info("Start model training")
    for epoch in range(1, EPOCHS_NUM + 1):
        model.train(mode=True)

        # model.hidden = model.init_hidden(BATCH_SIZE)
        for i, (x_batch, y_batch) in tqdm(enumerate(trn_dl), total=len(trn_dl), desc="Training epoch: "):
            if x_batch.size(0) != BATCH_SIZE:
                continue

            # model.init_hidden(x_batch.size(0))
            x_batch, y_batch = x_batch.double().to(device), y_batch.double().to(device)

            print("x_batch: ", x_batch.shape)
            print("y_batch: ", y_batch.shape)

            # Forward pass
            out = model(x_batch)

            print("out: ", out)
            print("y_batch: ", y_batch)

            print("out: ", out.shape)
            # out = out.view(out.size(0) * out.size(1), out.size(2))
            # y_batch = y_batch.view(y_batch.size(0) * y_batch.size(1))

            # print("out: ", out.shape)

            loss = criterion(out, y_batch.long())
            # loss = criterion(out.view(out.size(0) * out.size(1), out.size(2)), y_batch.view(y_batch.size(0) * y_batch.size(1)).long())

            # zero the parameter gradients
            optimizer.zero_grad()

            # Backward and optimize
            loss.backward()
            optimizer.step()

        val_loss, val_correct, val_total, val_acc = get_train_metric(model, val_dl, criterion, BATCH_SIZE)
        train_loss, train_correct, train_total, train_acc = get_train_metric(model, trn_dl, criterion, BATCH_SIZE)

        logging.info(f'Epoch: {epoch:3d} |'
                     f' Validation Loss: {val_loss:.2f}, Validation Acc.: {val_acc:2.2%}, '
                     f'Train Loss: {train_loss:.2f}, Train Acc.: {train_acc:2.2%}'
                     )

        if val_acc > best_acc:
            trials = 0
            best_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/simple_lstm_best.pth')
            logging.info(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                logging.info(f'Early stopping on epoch {epoch}')
                break

        scheduler.step(val_loss)


if __name__ == '__main__':
    train()
