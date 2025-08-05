import os
from tqdm import tqdm
import requests

def download_mnist_if_needed(data_dir='./data'):
    files = [
        ('train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'),
        ('train-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'),
        ('t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'),
        ('t10k-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'),
    ]
    os.makedirs(data_dir, exist_ok=True)
    for fname, url in files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname} ...")
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length', 0))
            with open(fpath, 'wb') as file, tqdm(
                desc=fname, total=total, unit='B', unit_scale=True
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
        else:
            print(f"{fname} already exists.")

# 在主脚本中调用
# from util import download_mnist_if_needed
# download_mnist_if_needed('./data')