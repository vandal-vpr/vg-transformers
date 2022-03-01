import os
import shutil
from tqdm import tqdm
from os.path import join as pjoin
import socket


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


moves = {
    ('train', 'test'): [],
    ('train', 'val'): ['amsterdam', 'manila'],
    # ('val', 'test'): ['cph', 'sf'],
}
import sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(script_dir))
from tvg.utils import get_all_datasets_path
DS_PATH = get_all_datasets_path()

create_split_dir('train')
create_split_dir('val')
create_split_dir('test')

for move in moves:
    src = move[0]
    dst = move[1]
    cities = moves[move]

    for city in cities:

        city_db_seqs = get_city_seqs(src, 'database', city)
        city_q_seqs = get_city_seqs(src, 'queries', city)

        move_seqs(src, dst, 'database', city_db_seqs)
        move_seqs(src, dst, 'queries', city_q_seqs)
