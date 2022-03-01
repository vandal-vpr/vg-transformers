import os
import shutil


DATASET_PATH = '/home/gabrielet/datasets/robotcar_cut_2m'
OUTPUT_PATH = '/home/gabrielet/datasets/robotcar_cut_2m_formatted'

train_dates = {'query': ['2015-03-17-11-08-44'], 'db': ['2014-12-16-18-44-24']}
val_dates = {'query': ['2014-11-18-13-20-12'], 'db': ['2014-11-21-16-07-03']}
test_dates = {'query': ['2015-07-08-13-37-17'], 'db': ['2014-11-14-16-34-33']}


def copy_img_list(images, base_src, base_dst):
    for image in images:
        shutil.copy(os.path.join(base_src, image), os.path.join(base_dst, image))


def format_set(type_set, dates):
    print(f'---Formatting {type_set} set with dates: {dates}---')
    os.makedirs(os.path.join(OUTPUT_PATH, type_set), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, type_set, 'queries'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, type_set, 'database'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, type_set, 'queries', 'sequence'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, type_set, 'database', 'sequence'), exist_ok=True)

    db_images = []
    for db_date in dates['db']:
        db_images += os.listdir(os.path.join(DATASET_PATH, db_date))
        copy_img_list(db_images, os.path.join(DATASET_PATH, db_date),
                      os.path.join(OUTPUT_PATH, type_set, 'database', 'sequence'))

    q_images = []
    for q_date in dates['query']:
        q_images += os.listdir(os.path.join(DATASET_PATH, q_date))
        copy_img_list(q_images, os.path.join(DATASET_PATH, q_date),
                      os.path.join(OUTPUT_PATH, type_set, 'queries', 'sequence'))


if __name__ == '__main__':
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    format_set('train', train_dates)
    format_set('val', val_dates)
    format_set('test', test_dates)
