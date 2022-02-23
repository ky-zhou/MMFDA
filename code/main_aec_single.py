import torch
import os
from options import MODEL_DIR, opt
from data_provider import *
from singlemodal import SMA
import sklearn.metrics as sk


if __name__ == '__main__':
    if not os.path.exists(os.path.join(MODEL_DIR, 'code/')):
        os.makedirs(os.path.join(MODEL_DIR, 'code/'))
        os.system('cp -r * %s' % (os.path.join(MODEL_DIR, 'code/')))  # bkp of model_ae1 def

    """direct training"""
    """settings"""
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    """parameter configuration"""
    m1, m2, m3, sample, label = load_csv(opt.input_type)
    m_use = m3
    single_m = torch.tensor(m_use).float().to(device)
    label = torch.tensor(label).long().to(device)
    in_dims = single_m.shape[1]
    savepath = MODEL_DIR + "/model-sma.h5"
    x_train = single_m[:-opt.test_size]
    sample_train, label_train = sample[:-opt.test_size], label[:-opt.test_size]
    x_test = single_m[-opt.test_size:]
    sample_test, label_test = sample[-opt.test_size:], label[-opt.test_size:]
    """autoencoder"""
    ae = SMA(in_dims, num_views=1, pred_missing=False).float().to(device)
    ae.fit(x_train, label_train, path=savepath, num_epochs=opt.max_epoch, lr=0.001)
    y_logit, hid = ae.predict(x_test, label_test, path=savepath)
    label_pred = y_logit.cpu().detach().numpy().argmax(1)
    y_logit = y_logit.cpu().detach().numpy()[:, 1]
    y_gt = label_test.cpu().detach().numpy()
    balanced_acc = sk.balanced_accuracy_score(y_gt, label_pred)
    acc = sk.accuracy_score(y_gt, label_pred)
    auc = sk.roc_auc_score(y_gt, y_logit)
    print('Balanced Acc: %.6f, AUC: %.6f.' % (balanced_acc, auc))


