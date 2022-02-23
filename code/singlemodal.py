import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import accuracy_score as calc_acc


class SMA(nn.Module):
    def __init__(self, in_dims, num_views, pred_missing=True):  # input2 is unavailable in testing
        super(SMA, self).__init__()
        self.num_views = num_views
        self.pred_missing = pred_missing

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.CrossEntropyLoss()
        hid1, hid2, hid3, hid_dec = 1024, 256, 12, 512
        self.encoder = nn.Sequential(
                nn.Linear(in_dims, hid1),
                nn.ReLU(),
                nn.BatchNorm1d(hid1, affine=True),
                nn.Linear(hid1, hid2),
                nn.ReLU(),
                nn.BatchNorm1d(hid2, affine=True),
                nn.Linear(hid2, hid3),
                nn.ReLU(),
                )
        self.decoder = nn.Sequential(
                nn.Linear(hid3, hid_dec),
                nn.ReLU(),
                nn.Linear(hid_dec, in_dims)
        )

        self.classi_layer = nn.Sequential(nn.Linear(hid3, 2), nn.Sigmoid())

    def forward(self, x):
            z = self.encoder(x)
            y = self.classi_layer(z)
            x_pred = self.decoder(z)
            return x_pred, z, y

    def loss_func(self, x_pred, x, y_pred, y):
        loss_mse = self.loss_mse(x_pred, x)
        loss_bce = self.loss_bce(y_pred, y)
        loss = loss_mse + loss_bce
        return loss

    def fit(self, x, y, path, num_epochs=200, lr=1e-3):
        print("Start training...")
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}], weight_decay=1e-5)
        self.train()
        for epoch in range(num_epochs):
            x_pred, hids, y_pred = self.forward(x)  # b, n (number of cluster)
            loss = self.loss_func(x_pred, x, y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct = (y_pred.argmax(1) == y).float().sum().item()
            accuracy = correct / len(y_pred)
            print('Epoch {}: Loss: {:6f}. Acc: {:6f}.'.format(epoch, loss.item(), accuracy))

            if epoch == num_epochs - 1:
                self.save_model(path)
                print('Training done...')

    def predict(self, x, y, path):
        self.load_model(path)
        self.eval()
        x_pred, hids, y_pred = self.forward(x)  # b, n (number of cluster)
        loss = self.loss_func(x_pred, x, y_pred, y)
        print('Loss: ', loss.item())
        return y_pred, hids

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
