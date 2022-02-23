import numpy as np
import pandas as pd
from options import MODEL_DIR, opt, Debug_Index
from sklearn.preprocessing import normalize
import pyreadr
test_size = opt.test_size
a = 0


def load_csv(which):
    gedf = pd.read_csv('../data/%s/%s-ge.csv' % (which, which), sep=",").to_numpy()
    medf = pd.read_csv('../data/%s/%s-me.csv' % (which, which), sep=",").to_numpy()
    midf = pd.read_csv('../data/%s/%s-mi.csv' % (which, which), sep=",").to_numpy()
    sample, ge, me, mi = gedf[:, 0], gedf[:, 1:], medf[:, 1:], midf[:, 1:]
    ge, me, mi = normalize(X=ge, axis=a, norm="max"), normalize(X=me, axis=a, norm="max"), normalize(X=mi, axis=a, norm="max")
    label_info = pd.read_csv('../data/%s/%s-label.csv' % (which, which), header=None).to_numpy()
    label = label_info[:, 1].astype(np.int32) - 1
    indices = np.arange(sample.shape[0])
    np.random.seed(opt.seed)
    np.random.shuffle(indices)
    ge, me, mi = ge[indices], me[indices], mi[indices]
    sample, label = sample[indices], label[indices]
    label0, label1 = sum([1 for x in label if x==0]), sum([1 for x in label if x==1])
    print('Data dimensions: ', ge.shape, me.shape, mi.shape, label0, label1)
    return ge, me, mi, sample, label



if __name__ == '__main__':
    # tuple = load_h5("../data_process/snn_data.h5")
    # x, y, s1, s2 = get_batch(tuple, False)
    load_csv(opt.input_type)
    # load_geo('gbm')
    # load_raw()
    # save_sample()
    # load_rdata(opt.input_type)
    # gen_rdata(opt.input_type)
    # gen_csv(opt.input_type)
    # load_rdata4cluster(opt.input_type)
    # load_pr_processed(opt.input_type)
    # load_pr(opt.input_type)
    # calc_pins_label(opt.input_type)
    # define_label(opt.input_type)

