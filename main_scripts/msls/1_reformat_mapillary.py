import os
import sys
import shutil
from glob import glob
from tqdm import tqdm
from os.path import join, dirname, isfile
import pandas as pd
import utm
import re
import argparse

parser = argparse.ArgumentParser(description="""Script to reformat the tree structure of MSLS dataset. Note that the 
dataset will be original dataset will be copied in the new structure without deleting it. If you wish to make the process
faster, and do not mind about deleting the original version of MSLS, you can specify --delete_old, that will `mv` 
rather than copy""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('root', type=str,
                    help='root directory of the original MSLS. Refer to the MSLS website for downloading it')
parser.add_argument('dest', type=str,
                    help='This is the directory in which the new version of MSLS will be created.')
parser.add_argument("--delete_old", action='store_true', default=False,
                    help='If you specify this option, instead of duplicating the dataset it will only change the tree structure')

def get_dst_image_name(latitude, longitude, pano_id='', timestamp='', city='',
                       seq_id='', frame_num='', extension=".jpg"):
    easting, northing, _, _, _, _ = format_location_info(latitude, longitude)

    assert is_valid_timestamp(timestamp), f"{timestamp} is not in YYYYMMDD_hhmmss format"

    return f"@{easting}@{northing}@{seq_id}@{frame_num}" + \
           f"@{pano_id}@{city}@{timestamp}@{extension}"


def format_location_info(latitude, longitude):
    easting, northing, zone_number, zone_letter = utm.from_latlon(float(latitude), float(longitude))
    easting = format_coord(easting, 7, 2)
    northing = format_coord(northing, 7, 2)
    latitude = format_coord(latitude, 3, 5)
    longitude = format_coord(longitude, 4, 5)
    return easting, northing, zone_number, zone_letter, latitude, longitude


def is_valid_timestamp(timestamp):
    """Return True if it's a valid timestamp, in format YYYYMMDD_hhmmss,
        with all fields from left to right optional.
    >>> is_valid_timestamp('')
    True
    >>> is_valid_timestamp('201901')
    True
    >>> is_valid_timestamp('20190101_123000')
    True
    """
    return bool(re.match("^(\d{4}(\d{2}(\d{2}(_(\d{2})(\d{2})?(\d{2})?)?)?)?)?$", timestamp))


def format_coord(num, left=2, right=5):
    """Return the formatted number as a string with (left) int digits
            (including sign '-' for negatives) and (right) float digits.
    >>> format_coord(1.1, 3, 3)
    '001.100'
    >>> format_coord(-0.123, 3, 3)
    '-00.123'
    """
    sign = "-" if float(num) < 0 else ""
    num = str(abs(float(num))) + "."
    integer, decimal = num.split(".")[:2]
    left -= len(sign)
    return f"{sign}{int(integer):0{left}d}.{decimal[:right]:<0{right}}"


default_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}


if __name__ == '__main__':
    args = parser.parse_args()

    root = args.root
    base_dst_folder = args.dest

    if not os.path.isdir(root):
        raise ValueError(f'The folder {root} does not exist')
    if not os.path.isdir(base_dst_folder):
        raise ValueError(f'The folder {base_dst_folder} does not exist')

    csv_files_paths = sorted(glob(join(root, "*", "*", "*", "postprocessed.csv"), recursive=True))
    for csv_file_path in csv_files_paths:
        with open(csv_file_path, "r") as file:
            postprocessed_lines = file.readlines()[1:]
        with open(csv_file_path.replace("postprocessed", "raw"), "r") as file:
            raw_lines = file.readlines()[1:]
        assert len(raw_lines) == len(postprocessed_lines)

        seq_file = (join(dirname(csv_file_path), 'seq_info.csv'))
        csv_dir = os.path.dirname(csv_file_path)
        city_path, folder = os.path.split(csv_dir)
        city = os.path.split(city_path)[1]

        seq_df = pd.read_csv(seq_file, index_col=[0])
        seq_mappings = {}
        for _, row in seq_df.iterrows():
            """
            if row.sequence_key in seq_mappings:
                seq_mappings[row.key].append((row.sequence_key, row.frame_number))
            else:"""
            seq_mappings[row.key] = (row.sequence_key, row.frame_number)

        folder = "database" if folder == "database" else "queries"
        train_val = "train" if city in default_cities["train"] else "val"
        dst_folder = os.path.join(base_dst_folder, train_val, folder)

        os.makedirs(dst_folder, exist_ok=True)
        for postprocessed_line, raw_line in zip(tqdm(postprocessed_lines, desc=city), raw_lines):
            _, pano_id, lon, lat, _, timestamp, is_panorama = raw_line.split(",")
            if is_panorama == "True\n":
                continue
            timestamp = timestamp.replace("-", "")

            seq_id = seq_mappings[pano_id][0]
            frame_num = seq_mappings[pano_id][1]
            dst_image_name = get_dst_image_name(lat, lon, pano_id, timestamp=timestamp, city=city,
                                                seq_id=seq_id, frame_num=frame_num)

            seq_folder = join(dst_folder, seq_id)
            os.makedirs(seq_folder, exist_ok=True)

            src_image_path = os.path.join(os.path.dirname(csv_file_path), 'images', f'{pano_id}.jpg')
            dst_image_path = os.path.join(seq_folder, dst_image_name)
            if not args.delete_old:
                _ = shutil.copy(src_image_path, dst_image_path)
            else:
                _ = shutil.move(src_image_path, dst_image_path)
