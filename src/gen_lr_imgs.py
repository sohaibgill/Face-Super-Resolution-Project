import cv2
from glob import glob
from tqdm import tqdm
import random
from threadpool import ThreadPool, makeRequests
import os

all_list = glob('../CelebA/img_celeba/*')
target_size = 64
# there are 3 quality ranges for each img
quality_ranges = [(15, 75)]
output_path = '../lr_64'
os.makedirs(output_path, exist_ok=True)


def saving(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    quality = random.randint(30, 75)
    img_path = output_path + '/' + path.rsplit('\\')[-1]
    cv2.imwrite(img_path, img,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality])


with tqdm(total=len(all_list), desc='Resizing images') as pbar:
    def callback(req, x):
        pbar.update()


    t_pool = ThreadPool(12)
    requests = makeRequests(saving, all_list, callback=callback)
    for req in requests:
        t_pool.putRequest(req)
    t_pool.wait()
