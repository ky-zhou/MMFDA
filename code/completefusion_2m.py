import torch.nn as nn
import torch


class CFA(nn.Module):
    def __init__(self, in_dims, num_views, pred_missing=True):  # input2 is unavailable in testing
        super(CFA, self).__init__()
        self.num_views = num_views
        self.pred_missing = pred_missing
        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.CrossEntropyLoss()
        hid1, hid2, hid3, hid_dec = 1024, 256, 12, 512
        enc = nn.Sequential(
            nn.Linear(in_dims[0], hid1),
            nn.ReLU(),
            nn.BatchNorm1d(hid1, affine=True),
            nn.Linear(hid1, hid2),
            nn.ReLU(),
            nn.BatchNorm1d(hid2, affine=True),)
        self.encoder.append(enc)
        enc = nn.Sequential(
            nn.Linear(in_dims[1], 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, affine=True),)
        self.encoder.append(enc)
        self.decoder.append(nn.Linear(hid_dec, in_dims[0]))
        self.decoder.append(nn.Linear(hid_dec, in_dims[1]))

        self.dec1 = nn.Sequential(nn.Linear(num_views*hid3, hid_dec), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hid2 + 64, num_views*hid3), nn.ReLU())
        self.classi_layer = nn.Sequential(nn.Linear(num_views*hid3, 2), nn.Sigmoid())

    def forward(self, x):
        x_preds, zs = [], []
        for i in range(self.num_views):
            z = self.encoder[i](x[i])
            zs.append(z)
        z = torch.cat(zs, 1)
        x_f = self.fusion(z) # num_views*hid3
        y = self.classi_layer(x_f)
        x_d1 = self.dec1(x_f)
        for i in range(self.num_views):
            x_pred = self.decoder[i](x_d1)
            x_preds.append(x_pred)
        return x_preds, zs, y

    def loss_func(self, x_pred, x, y_pred, y):
        loss_mse = []
        for i in range(self.num_views):
            loss_mse.append(self.loss_mse(x_pred[i], x[i]))
        loss_bce = self.loss_bce(y_pred, y)
        loss = torch.mean(torch.stack(loss_mse, 0)) + loss_bce
        return loss

    def fit(self, x, y, path, num_epochs=200, lr=1e-3):
        print("Start training...")
        optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': lr}], weight_decay=1e-5)
        self.train()
        for epoch in range(num_epochs):
            x_pred, hids, y_pred = self.forward(x)  # b, n (number of cluster)
            # print('y', y_pred.shape, y.shape)
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
