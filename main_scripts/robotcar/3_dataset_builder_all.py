from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import numpy as np
import os
from datetime import datetime as dt
import csv
import glob
from sklearn.neighbors import NearestNeighbors


def load_image(image_path):
    """Loads and rectifies an image from file."""
    camera = 'stereo'
    if camera == 'stereo':
        pattern = 'gbrg'
    else:
        pattern = 'rggb'
    img = Image.open(image_path)
    img = demosaic(img, pattern)
    return Image.fromarray(np.array(img).astype(np.uint8))


def path_dir_create_if_not_exists(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def retrieve_gps(path_gps, src_images, dst_images, current, lap_date):
    path_dir_create_if_not_exists(dst_images)
    print(f"src_images: {src_images}, dst_images: {dst_images}")
    dict_gps = {}
    with open(path_gps, mode='r') as csv_file:
        rows = list(csv.DictReader(csv_file))

    for row in rows:  # header row is already skipped automatically
        timestamp = int(row["timestamp"])
        dict_gps[timestamp] = (row["northing"], row["easting"])

    print(f'Processed {len(dict_gps)} gps coordinates')
    gps_timestamps = np.array(list(dict_gps.keys())).reshape(-1, 1)

    # the image name is a timestamp, so this is equivalent to sort timestamps
    images_paths = sorted(os.listdir(src_images))
    images_timestamps = np.array(list(map(lambda p: int(p.split(".")[0]), images_paths))).reshape(-1, 1)

    # a ogni timestamp in cui è stata scattata la foto devo associare il 
    # timestamp più vicino a cui è stato ottenuto il gps
    print(
        f"Calcolo i KNN (k=1) per i timestamps di {len(images_timestamps)} immagini su {len(gps_timestamps)} coordinate gps")
    knn = NearestNeighbors()
    knn.fit(gps_timestamps)
    nearest_gps_timestamps = knn.kneighbors(images_timestamps, 1, return_distance=False)
    print("Finito il calcolo della KNN")

    for image_index, image_path in enumerate(images_paths):
        nearest_gps_timestamp = gps_timestamps[nearest_gps_timestamps[image_index]][0, 0]
        lat, lng = dict_gps[nearest_gps_timestamp]
        image_timestamp = images_timestamps[image_index, 0]
        # new_name = f"@{lat}@{lng}@{str(dt.utcfromtimestamp(image_timestamp/1000000))[11:]}@{image_timestamp}@.jpg"
        new_name = f"@{lat}@{lng}@{lap_date}@{image_index}@{image_timestamp}@.jpg"
        new_name = os.path.join(dst_images, new_name)
        img = load_image(os.path.join(src_images, image_path))
        img.save(new_name)
        if image_index % 500 == 0:
            print(
                f"{str(dt.now())[11:19]}   {current} [{image_index:05d}/{len(images_paths)}] Saving {image_path} to {new_name}")


DATASET_PATH = '/home/gabrielet/datasets/robotcar_val'
OUTPUT_PATH = '/home/gabrielet/datasets/robotcar_val_rgb'

if __name__ == '__main__':
    datasets = [ds for ds in os.listdir(DATASET_PATH) if ds.startswith('2014') or ds.startswith('2015')]

    for dataset_index, dataset in enumerate(datasets):
        path_metadata = os.path.join(DATASET_PATH, 'meta', dataset, 'rtk.csv')
        src_images = os.path.join(DATASET_PATH, dataset, 'stereo', 'left')
        dst_images = os.path.join(OUTPUT_PATH, dataset)
        current = f"{dataset_index + 1:02d}/{len(datasets):02d}"
        retrieve_gps(path_metadata, src_images, dst_images, current, dataset)
