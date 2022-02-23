import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_type', default='gbm', help='gbm, laml, paad')
parser.add_argument('--index', type=int, default=1, help='model index')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=200, help='br 72, gbm 73, la 64, pr 495')
parser.add_argument('--test_size', type=int, default=100, help='gbm 100, laml 143, paad 175||0.36')
parser.add_argument('--seed', type=int, default=1, help='0')
parser.add_argument('--pred_missing', nargs='?', type=bool, const=True, default=False, help='predict missing modality?')
opt = parser.parse_args()

BASE_DIR = os.getcwd()
Debug_Index = str(opt.index)
MODEL_DIR = os.path.join('../model', Debug_Index)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

