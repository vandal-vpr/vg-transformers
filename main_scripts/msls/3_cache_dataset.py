from init_scripts import init_script
init_script()
import argparse
from tvg.datasets import cache_mapillary_train, retrieve_db_object
import re

parser = argparse.ArgumentParser(description="cache train dataset object")
parser.add_argument('--seq', type=int, default=3)
parser.add_argument("--dataset_path", type=str, default='', help="Path of the dataset")
parser.add_argument('--save_to', type=str, default='', help='path to save the object')
args = parser.parse_args()

folder = args.save_to

seq_len = args.seq
nNeg = 5
cities = ''
f_cities = cities if cities == '' else ('_' + cities)
print(f"caching with seq lenght : {seq_len}...")
if seq_len % 2 == 0:
    cut_last_frame = True
    seq_len = seq_len + 1
    print(f"Even sequence --> Seq_len {seq_len} and cut_last_frame {cut_last_frame}")
else:
    cut_last_frame = False
cache_mapillary_train(args.dataset_path,
                      f'{folder}/reformat_msls{f_cities}_train_{seq_len}seq_{nNeg}neg.pkl',
                      split='train',
                      posDistThr=10,
                      negDistThr=25,
                      nNeg=nNeg,
                      cities=cities,
                      seq_length=seq_len,
                      cut_last_frame=cut_last_frame)
