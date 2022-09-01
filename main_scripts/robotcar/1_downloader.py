import requests
import os
from lxml import html
from datetime import datetime
import re


def count_files(date, prefix='_stereo_left_', camera='Bumblebee XB3', following_camera='Grasshopper 2 Left'):
    with requests.Session() as session_requests:
        result = session_requests.get(f'https://robotcar-dataset.robots.ox.ac.uk/datasets/{date}')
        tree = html.fromstring(result.text)

    start = result.text.find(camera)
    end = result.text.find(following_camera)
    files = re.findall(f'{prefix}0..tar</a>', result.text[start:end])

    return len(files)


def read_credentials(secret_file):
    with open(secret_file, 'r') as f:
        creds = f.readline().split(',')
    user = creds[0].strip()
    passw = creds[1].strip()

    return user, passw


def login_and_download(log_url, url, dump_to):
    user, passw = read_credentials(SECRET_FILE)
    with requests.Session() as session_requests:
        result = session_requests.get(log_url)
        tree = html.fromstring(result.text)
        csrf_middleware_token = list(set(tree.xpath("//input[@name='csrfmiddlewaretoken']/@value")))[0]
        payload = {"username": user, "password": passw,
                   "csrfmiddlewaretoken": csrf_middleware_token}
        p = session_requests.post(login_url, data=payload, headers=dict(referer=login_url))

        response = session_requests.get(url)
        with open(os.path.join(out_dir, dump_to), 'wb') as f:
            f.write(response.content)


login_url = "https://mrgdatashare.robots.ox.ac.uk/"
SECRET_FILE = '' # obtain the credentials from the url and put them in a plain txt file
assert SECRET_FILE != '', 'you should set the path of the file containing credentials'

out_dir = os.path.join(os.path.abspath(os.curdir), 'downloaded')
os.makedirs(out_dir, exist_ok=True)
PREFIX = '_stereo_left_'
dates = [
    ['2014-12-17-18-18-43', '2014-12-16-09-14-09'], # train: (query: winter rain night, db: winter day)
    ['2015-02-03-08-45-10', '2015-11-13-10-28-08'], # val: (query: winter snow day, db: fall overcast day)
    ['2014-12-16-18-44-24', '2014-11-18-13-20-12'], # test: (query: winter night, db: fall day)
]

for date_set in dates:
    for date in dates:
        n_files = count_files(date=date, prefix=PREFIX)
        filenames = [date + PREFIX + '0' + str(i) + '.tar' for i in range(1, n_files + 1)]

        meta_url = 'http://mrgdatashare.robots.ox.ac.uk:80/download/?filename=rtk.zip'
        print(f"{str(datetime.now())[:19]}   "
              f"    Sto scaricando i metadati")
        login_and_download(login_url, meta_url, 'meta.zip')

        for i, filename in enumerate(filenames):
            url = f'http://mrgdatashare.robots.ox.ac.uk:80/download/?filename=datasets/{date}/{filename}'
            print(f"{str(datetime.now())[:19]}   "
                  f"{i + 1}/{len(filenames)}    Sto scaricando {filename}")
            login_and_download(login_url, url, filename)
