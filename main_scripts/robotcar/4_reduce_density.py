import math
import shutil
import os
from glob import glob
import numpy as np


def get_dist(northing1, easting1, northing2, easting2):
    return math.sqrt((northing2-northing1)**2 + (easting2-easting1)**2)


def renumber_frame(image_name, new_number):
    info = image_name.split('@')
    new_name = f"@{info[fields['northing']]}@{info[fields['easting']]}@{info[fields['date']]}@{new_number}@{info[fields['timestamp']]}@.jpg"

    return new_name


MIN_DIST = 2 # in meters between frames
DATASET_PATH = '/home/gabrielet/datasets/robotcar_val_rgb'
OUTPUT_PATH = f'/home/gabrielet/datasets/robotcar_val_cut_{MIN_DIST}m'
fields = {
    'northing': 1,
    'easting': 2,
    'date': 3,
    'frame_num': 4,
    'timestamp': 5
}

if __name__ == '__main__':
    dates = [ds for ds in os.listdir(DATASET_PATH) if ds.startswith('2014') or ds.startswith('2015')]

    for date in dates:
        print(f'Lap {date}:')
        image_names = os.listdir(os.path.join(DATASET_PATH, date))
        timestamps_order = np.argsort([image_name.split('@')[fields['timestamp']] for image_name in image_names])
        utms = np.array([(float(image_name.split('@')[fields['northing']]), float(image_name.split('@')[fields['easting']]))
                         for image_name in image_names])
        utms_sorted = [tuple(utm) for utm in utms[timestamps_order]]
        filenames_sorted = list(np.array(image_names)[timestamps_order])

        utms_to_filename = {utm: filenames_sorted[i] for i, utm in enumerate(utms_sorted)}
        good_lines = [utms_sorted[0]]
        for i, utm in enumerate(utms_sorted[1:]):
            last_good = good_lines[-1]
            dist = get_dist(*last_good, *utm)
            if dist >= MIN_DIST:
                good_lines.append(utm)

        print(f'\tKeeping {len(good_lines)} good images over {len(image_names)} total')
        good_image_names = [utms_to_filename[gl] for gl in good_lines]

        os.makedirs(os.path.join(OUTPUT_PATH, date), exist_ok=True)
        for i, good_image in enumerate(good_image_names):
            new_img_name = renumber_frame(good_image, i)
            shutil.copy(os.path.join(DATASET_PATH, date, good_image), os.path.join(OUTPUT_PATH, date, new_img_name))
