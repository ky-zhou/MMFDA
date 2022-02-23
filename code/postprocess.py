import argparse
import numpy as np
import matplotlib as mpl
from data_provider2 import *
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from options import MODEL_DIR, opt

mpl.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=100, help='Epoch to run [default: 500]')
args = parser.parse_args()


def tsne(f1, f2, fig_name):
    from time import time
    from sklearn import manifold
    t0 = time()

    X = np.loadtxt(f1, delimiter=",")
    label = np.loadtxt(f2, delimiter=",").astype(np.int32)
    n_components = 2
    color_map = {0: 'lime', 1: 'red'}
    marker_map = {0: 'o', 1: '^'}
    label_map = {0: 'label 0', 1: 'label 1'}

    method = manifold.TSNE(n_components=n_components, init='random', random_state=0)
    Y = method.fit_transform(X)
    fig = plt.figure(figsize=(4, 4))
    # ax = fig.add_subplot(4, 4, 4)
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plt.title("T-SNE of Testing Features", fontsize=32)
    for i in range(2):
        plt.scatter(Y[label == i][:, 0], Y[label == i][:, 1], s=48, c=color_map[i],
                    label=label_map[i], marker=marker_map[i], cmap=plt.cm.rainbow)
    plt.axis('tight')
    plt.xlabel('t-SNE dimension 1', fontsize=6)
    plt.ylabel('t-SNE dimension 2', fontsize=6)
    plt.axis('off')
    fig.savefig(fig_name, bbox_inches='tight')
    t1 = time()
    print("TSNE processed in: %.2g sec" % (t1 - t0))


def canonical():
    # m1, m2, m3, sample, label = load_rdata(opt.input_type)
    m1, m2, m3, sample, label = load_csv(opt.input_type)
    x, y = m1, m2
    plsca = PLSCanonical(n_components=2)
    plsca.fit(x, y)
    x_t, y_t = plsca.transform(x, y)
    print('dimension reduction:', x_t.shape, y_t.shape)

    fig = plt.figure(figsize=(6, 4))
    plt.scatter(x_t[:, 0], y_t[:, 0], label="Dim 1", marker="o", s=25)
    plt.scatter(x_t[:, 1], y_t[:, 1], label="Dim 2", marker="x", s=25)
    plt.title('$corr_{dim1}$ = %.2f, $corr_{dim2}$ = %.2f'
              % (np.corrcoef(x_t[:, 0], y_t[:, 0])[0, 1], np.corrcoef(x_t[:, 1], y_t[:, 1])[0, 1]))
    plt.legend(loc=4)
    # plt.axis('off')
    """do not show axis values"""
    fig.axes[0].xaxis.set_ticklabels([])
    fig.axes[0].yaxis.set_ticklabels([])
    plt.xlabel("$m_1$", size=16)
    plt.ylabel("$m_2$", size=16)
    fig.savefig('../canonical-%s.png'%opt.input_type, bbox_inches='tight')


def calc_acc(files, fig_name):
    import matplotlib.cm as cm
    # shap_file, exp_file, deg_file, non_de21file, random_file, noise
    plt.rcParams.update({'font.size': 12})
    data, labels = [], ['CFA', 'IFA', 'CFA-2M', 'SMA-$m_1$', 'SMA-$m_2$', 'SMA-$m_3$']
    color_box = ['greenyellow', 'cornflowerblue', 'lightcoral', 'lightblue', 'bisque', 'lightgreen']
    # data, labels = [], ['CFA', 'IFA', 'CFA-2M', 'SMA-$m_1$', 'SMA-$m_2$', 'SMA-$m_3$', 'CFA*', 'IFA*']
    # color_box = ['greenyellow', 'cornflowerblue', 'lightcoral', 'lightblue', 'bisque', 'lightgreen', 'greenyellow', 'cornflowerblue']
    for file in files:
        acc = np.loadtxt(file).astype(np.float32)
        data.append(acc)
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    medianprops = dict(linestyle='--', linewidth=2.5, color='firebrick')
    accs = np.array(data)
    print(accs.mean(1), accs.shape)
    bplot = ax1.boxplot(data, patch_artist=True, labels=labels, medianprops=medianprops, widths=0.8)
    # bplot = ax1.boxplot(data, notch=True, patch_artist=True, labels=labels, widths=0.8)
    for patch, color in zip(bplot['boxes'], color_box):
        patch.set_facecolor(color)
    fig1.savefig(fig_name)


def roc_plot(file_list, fig_name, show_fill=True, pfi=False):
    import sklearn.metrics as sk
    alpha, lw = 0.7, 2
    color_box = ['greenyellow', 'cornflowerblue', 'lightcoral', 'lightblue', 'bisque', 'lightgreen', ]#, 'dimgray', 'black']
    curve_comment = ['Avg ROC of CFA (AUC=%0.4f)', 'Avg ROC of IFA (AUC=%0.4f)', 'Avg ROC of CFA-2M (AUC=%0.4f)',
                     'Avg ROC of SMA-mRNA (AUC=%0.4f)', 'Avg ROC of SMA-methylation (AUC=%0.4f)',
                     'Avg ROC of SMA-miRNA (AUC=%0.4f)',
                     # 'Avg ROC of CFA* (AUC=%0.4f)', 'Avg ROC of IFA* (AUC=%0.4f)',
                     ]
    fig = plt.figure(figsize=(6, 6))
    base_fpr = np.linspace(0, 1, 101)
    for p, file in enumerate(file_list):
        # print("Processing file:", file)
        auc_avg, fprs, tprs = [], [], []
        upper, lower = np.zeros((len(base_fpr))), np.zeros((len(base_fpr)))
        with open(file) as f:
            lines = f.readlines()
            num_line = len(lines)
            for i in range(0, num_line, 3):
                fpr = lines[i+1].split(' ')
                tpr = lines[i+2].split(' ')
                tpr = [float(x) for x in tpr]
                fpr = [float(x) for x in fpr]
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0.0
                tprs.append(tpr)
            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)
            area = sk.auc(base_fpr, mean_tprs)
            for j, y in enumerate(mean_tprs):
                upper[j], lower[j] = mean_tprs[j]+tprs[:, j].std(), mean_tprs[j]-tprs[:, j].std()
            plt.plot(base_fpr, mean_tprs, color_box[p], linewidth=lw, label=curve_comment[p] % area)
    # plt.plot([0.0, 1.0], [0.0, 1.0], 'darkorange', ls=':', linewidth=lw, label='Noise')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc=4)
    fig.savefig(fig_name)


def all_tsne(plot_list, input_type):
    tsne('../model/%d/hid.csv'%plot_list[0], '../model/%d/testlabel.csv'%plot_list[0], '../tsne-%s-cfa.png'%input_type)
    tsne('../model/%d/hid.csv'%plot_list[1], '../model/%d/testlabel.csv'%plot_list[1], '../tsne-%s-ifa.png'%input_type)
    tsne('../model/%d/hid.csv'%plot_list[2], '../model/%d/testlabel.csv'%plot_list[2], '../tsne-%s-2m.png'%input_type)
    tsne('../model/%d/hid.csv'%plot_list[3], '../model/%d/testlabel.csv'%plot_list[3], '../tsne-%s-m1.png'%input_type)
    tsne('../model/%d/hid.csv'%plot_list[4], '../model/%d/testlabel.csv'%plot_list[4], '../tsne-%s-m2.png'%input_type)
    tsne('../model/%d/hid.csv'%plot_list[5], '../model/%d/testlabel.csv'%plot_list[5], '../tsne-%s-m3.png'%input_type)


def all_metrics(plot_list, input_type):
    calc_acc(['../model/%d/log_blacc.txt'%plot_list[0], '../model/%d/log_blacc.txt'%plot_list[1],
              '../model/%d/log_blacc.txt'%plot_list[2], '../model/%d/log_blacc.txt'%plot_list[3],
              '../model/%d/log_blacc.txt'%plot_list[4], '../model/%d/log_blacc.txt'%plot_list[5],
              ], "../blacc-%s-enc.png"%input_type)
    roc_plot(['../model/%d/log_auc.txt'%plot_list[0], '../model/%d/log_auc.txt'%plot_list[1],
              '../model/%d/log_auc.txt'%plot_list[2], '../model/%d/log_auc.txt'%plot_list[3],
              '../model/%d/log_auc.txt'%plot_list[4], '../model/%d/log_auc.txt'%plot_list[5],
              ], "../roc-%s-enc.png"%input_type)


def sh():
    for i in range(100):
        print('python main_aec.py --seed %d' % (i+1))


def statistic():
    from scipy.stats import ttest_ind, ks_2samp
    import seaborn as sns
    forpv = []
    for i in (6, 11):
        acc = np.loadtxt('../model/%d/log_blacc.txt'%(i)).astype(np.float32)*100
        print('median:', np.mean(acc))
        sns.set(style="ticks")
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(acc, ax=ax_box)
        sns.distplot(acc, ax=ax_hist)
        ax_box.set(yticks=[])
        sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box, left=True)
        fig.savefig('../tt-%s-%d.png'%(opt.input_type, i))
        forpv.append(acc)
    t, p = ttest_ind(forpv[0], forpv[1])
    res = ks_2samp(forpv[0], forpv[1])
    print(p, res)


def mut(which):
    import pandas as pd
    from adjustText import adjust_text
    import math
    df = pd.read_csv('../post/mutation-%s.csv' % (which), sep=",").to_numpy()
    gene, cord = df[:, 0], df[:, 1:]
    fig = plt.figure(figsize=(6, 4))
    new_list = range(math.floor(min(cord[:, 0])), math.ceil(max(cord[:, 0])) + 1)
    plt.yticks(new_list)
    plt.scatter(cord[:, 1], cord[:, 0], marker="*", )
    """add name to some points"""
    txt = []
    for i, c in enumerate(cord):
        if c[0] > 2 or c[1] > 100:
            print('x: ', gene[i])
            txt.append((gene[i], c))
    texts = [plt.text(t[1][1], t[1][0], t[0], size=12) for t in txt]
    adjust_text(texts, force_points=(1,1),force_text=(2.1,3.7), arrowprops=dict(arrowstyle='-', color='red', lw=1))
    plt.xlabel("Short-term survival", size=16)
    plt.ylabel("Long-term survival", size=16)
    fig.savefig('../mut-%s.png'%which, bbox_inches='tight')


def calc_acce(files, fig_name):
    import matplotlib.cm as cm
    # shap_file, exp_file, deg_file, non_de21file, random_file, noise
    plt.rcParams.update({'font.size': 12})
    # data, labels = [], ['IFA1', 'IFA2'
    #                     ]
    # color_box = ['cornflowerblue', 'blue'
    #              ]
    data, labels = [], ['CFA', 'IFA', 'CFA-2M', 'S$m_1$', 'S$m_3$',
                        'S`$m_1$', 'S`$m_3$', '2M`'
                        ]
    color_box = ['greenyellow', 'cornflowerblue', 'lightcoral', 'lightblue', 'lightgreen',
                 'blue', 'green', 'red'
                 ]
    # data, labels = [], ['CFA', 'IFA', 'CFA-2M', 'SMA-$m_1$', 'SMA-$m_2$', 'SMA-$m_3$', 'CFA*', 'IFA*']
    # color_box = ['greenyellow', 'cornflowerblue', 'lightcoral', 'lightblue', 'bisque', 'lightgreen', 'greenyellow', 'cornflowerblue']
    for file in files:
        acc = np.loadtxt(file).astype(np.float32)
        data.append(acc)
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    medianprops = dict(linestyle='--', linewidth=2.5, color='firebrick')
    accs = np.array(data)
    print(accs.mean(1), accs.shape)
    bplot = ax1.boxplot(data, patch_artist=True, labels=labels, medianprops=medianprops, widths=0.8)
    # bplot = ax1.boxplot(data, notch=True, patch_artist=True, labels=labels, widths=0.8)
    for patch, color in zip(bplot['boxes'], color_box):
        patch.set_facecolor(color)
    fig1.savefig(fig_name)


def roc_plote(file_list, fig_name, show_fill=True, pfi=False):
    import sklearn.metrics as sk
    alpha, lw = 0.7, 2
    # color_box = ['cornflowerblue', 'blue']#, 'dimgray', 'black']
    # curve_comment = ['Avg ROC of IFA1 (AUC=%0.4f)', 'Avg ROC of IFA2 (AUC=%0.4f)',
    #                ]
    color_box = ['greenyellow', 'cornflowerblue', 'lightcoral', 'lightblue', 'lightgreen',
                 'blue', 'green', 'red', ]#, 'dimgray', 'black']
    curve_comment = ['Avg ROC of CFA (AUC=%0.4f)', 'Avg ROC of IFA (AUC=%0.4f)', 'Avg ROC of CFA-2M (AUC=%0.4f)',
                     'Avg ROC of SMA-mRNA (AUC=%0.4f)', #'Avg ROC of SMA-methylation (AUC=%0.4f)',
                     'Avg ROC of SMA-miRNA (AUC=%0.4f)',
                     'Avg ROC of SMD-mRNA (AUC=%0.4f)', #'Avg ROC of SMD-methylation (AUC=%0.4f)',
                     'Avg ROC of SMD-miRNA (AUC=%0.4f)', 'Avg ROC of CFD-2M (AUC=%0.4f)',
                   ]
    fig = plt.figure(figsize=(6, 6))
    base_fpr = np.linspace(0, 1, 101)
    for p, file in enumerate(file_list):
        # print("Processing file:", file)
        auc_avg, fprs, tprs = [], [], []
        upper, lower = np.zeros((len(base_fpr))), np.zeros((len(base_fpr)))
        with open(file) as f:
            lines = f.readlines()
            num_line = len(lines)
            for i in range(0, num_line, 3):
                fpr = lines[i+1].split(' ')
                tpr = lines[i+2].split(' ')
                tpr = [float(x) for x in tpr]
                fpr = [float(x) for x in fpr]
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0.0
                tprs.append(tpr)
            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)
            area = sk.auc(base_fpr, mean_tprs)
            for j, y in enumerate(mean_tprs):
                upper[j], lower[j] = mean_tprs[j]+tprs[:, j].std(), mean_tprs[j]-tprs[:, j].std()
            plt.plot(base_fpr, mean_tprs, color_box[p], linewidth=lw, label=curve_comment[p] % area)
    # plt.plot([0.0, 1.0], [0.0, 1.0], 'darkorange', ls=':', linewidth=lw, label='Noise')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc=4)
    fig.savefig(fig_name)


def all_metricse(plot_list, input_type):
    # calc_acce(['../model/%d/log_blacc.txt'%plot_list[0], '../model/%d/log_blacc.txt'%plot_list[1],
    #           # '../model/%d/log_blacc.txt'%plot_list[2], '../model/%d/log_blacc.txt'%plot_list[3],
    #           # '../model/%d/log_blacc.txt'%plot_list[4], '../model/%d/log_blacc.txt'%plot_list[5],
    #            # '../model/%d/log_blacc.txt' % plot_list[6], #'../model/%d/log_blacc.txt' % plot_list[7],
    #            # '../model/%d/log_blacc.txt' % plot_list[8], '../model/%d/log_blacc.txt' % plot_list[9],
    #           ], "../blacc-%s-scale.png"%input_type)
    # roc_plote(['../model/%d/log_auc.txt'%plot_list[0], '../model/%d/log_auc.txt'%plot_list[1],
    #           # '../model/%d/log_auc.txt'%plot_list[2], '../model/%d/log_auc.txt'%plot_list[3],
    #           # '../model/%d/log_auc.txt'%plot_list[4], '../model/%d/log_auc.txt'%plot_list[5],
    #            # '../model/%d/log_auc.txt' % plot_list[6], #'../model/%d/log_auc.txt' % plot_list[7],
    #            # '../model/%d/log_auc.txt' % plot_list[8], '../model/%d/log_auc.txt' % plot_list[9],
    #           ], "../roc-%s-scale.png"%input_type)
    """encoder"""
    calc_acce(['../model/%d/log_blacc.txt'%plot_list[0], '../model/%d/log_blacc.txt'%plot_list[1],
              '../model/%d/log_blacc.txt'%plot_list[2], '../model/%d/log_blacc.txt'%plot_list[3],
              # '../model/%d/log_blacc.txt'%plot_list[4],
               '../model/%d/log_blacc.txt'%plot_list[5],
               '../model/%d/log_blacc.txt' % plot_list[6], #'../model/%d/log_blacc.txt' % plot_list[7],
               '../model/%d/log_blacc.txt' % plot_list[8], '../model/%d/log_blacc.txt' % plot_list[9],
              ], "../blacc-%s-enc.png"%input_type)
    roc_plote(['../model/%d/log_auc.txt'%plot_list[0], '../model/%d/log_auc.txt'%plot_list[1],
              '../model/%d/log_auc.txt'%plot_list[2], '../model/%d/log_auc.txt'%plot_list[3],
              # '../model/%d/log_auc.txt'%plot_list[4],
               '../model/%d/log_auc.txt'%plot_list[5],
               '../model/%d/log_auc.txt' % plot_list[6], #'../model/%d/log_auc.txt' % plot_list[7],
               '../model/%d/log_auc.txt' % plot_list[8], '../model/%d/log_auc.txt' % plot_list[9],
              ], "../roc-%s-enc.png"%input_type)


if __name__ == "__main__":
    # plot_list = [272, 271, 273, 274, 275, 276] #laml
    # plot_list = [152, 151, 153, 154, 155, 156] #paad shinked size
    plot_list = [120, 121, 122, 123, 124, 125, 13, 12, 14, 15] #paad shinked size
    # plot_list = [16, 17] #paad shinked size
    # all_tsne(plot_list, opt.input_type)
    # all_metrics(plot_list, opt.input_type)
    all_metricse(plot_list, opt.input_type)
    # canonical()
    # statistic() #271, 273
    # mut('gbm')
    # sh()