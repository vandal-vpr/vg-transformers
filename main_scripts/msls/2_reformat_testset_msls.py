import os
import shutil
from tqdm import tqdm
from os.path import join as pjoin
import socket
import argparse


parser = argparse.ArgumentParser(description="""Script to create val and test splits as they are proposed in our paper. 
The authors of MSLS never released the test set labels, so we decided to use the original validation set (`cph` and `sf`)
as test set, and to use as validation `amsterdam` and `manila` which have similar statistics to the test set. 
This script is meant to be executed AFTER `1_reformat_mapillary.py` 
""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('ds_folder', type=str,
                    help='root directory of the reformatted MSLS. It will create inside of it val and test splits')


def create_split_dir(split):
    os.makedirs(pjoin(DS_PATH, split), exist_ok=True)
    os.makedirs(pjoin(DS_PATH, split, 'queries'), exist_ok=True)
    os.makedirs(pjoin(DS_PATH, split, 'database'), exist_ok=True)


def get_city_for_sequence(base_path, sequence):
    image_name = os.listdir(pjoin(base_path, sequence))[0]
    return image_name.split('@')[6]


def get_city_seqs(split, subset, city):
    seqs_per_city = []
    base_path = pjoin(DS_PATH, split, subset)
    sequences = os.listdir(base_path)
    for seq in sequences:
        city_curr = get_city_for_sequence(base_path, seq)
        if city_curr in city:
            seqs_per_city.append(seq)
        """
        if seqs_per_city.get(city, None) is None:
            seqs_per_city[city] = [seq]
        else:
            seqs_per_city[city].append(seq)"""
    return seqs_per_city


def move_seqs(src_split, dst_split, subset, sequences):
    base_src = pjoin(DS_PATH, src_split, subset)
    base_dst = pjoin(DS_PATH, dst_split, subset)
    print(f'--Moving {city} {subset} seqs from {src_split} to {dst_split} split--')
    for seq in tqdm(sequences, ncols=100):
        shutil.move(pjoin(base_src, seq), pjoin(base_dst, seq))

# The dictionary below encodes the changes that will be made. From train to test nothing should be moved.
# From train, the cities of amsterdam and manila must be moved to the val split.
# Finally, the last line is commented, as in principle cph and sf should be moved from val to test,
# but since they are the entirety of the old validation set, to do that it is enough to rename the 'val'
# folder to 'test'
moves = {
    ('train', 'test'): [],
    ('train', 'val'): ['amsterdam', 'manila'],
    # ('val', 'test'): ['cph', 'sf'],
}

if __name__ == '__main__':
    args = parser.parse_args()

    DS_PATH = args.ds_folder
    if not os.path.isdir(DS_PATH):
        raise ValueError(f'The folder {root} does not exist')

    if os.path.isdir(pjoin(DS_PATH, 'test')):
        shutil.rmtree(pjoin(DS_PATH, 'test'))
    shutil.move(pjoin(DS_PATH, 'val'), pjoin(DS_PATH, 'test')) # -> this is doing : ('val', 'test'): ['cph', 'sf']

    create_split_dir('val')
    for move in moves:
        src = move[0]
        dst = move[1]
        cities = moves[move]

        for city in cities:

            city_db_seqs = get_city_seqs(src, 'database', city)
            city_q_seqs = get_city_seqs(src, 'queries', city)

            move_seqs(src, dst, 'database', city_db_seqs)
            move_seqs(src, dst, 'queries', city_q_seqs)
