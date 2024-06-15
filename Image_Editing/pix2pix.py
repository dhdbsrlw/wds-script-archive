import datasets
from multiprocessing import Pool
import os
import json
import webdataset as wds
from tqdm import tqdm
import math
import random

from PIL import Image
import base64
from io import BytesIO


DATASET_NAME = "pix2pix"
DATA_ROOT = "/data/instruct-pix2pix/pp_pix2pix.json" 
# For pix2pix, I preprocess the data to include [ folder_name, img_before, img_after, input, edit ] keys, because its raw data has complicated structure.
OUTPUT_DIR = f'/data/{DATASET_NAME}_wds'

CHUNK_SIZE = 10000


def convert(sub_ds, pid):

    n_samples = 0
    image_root_prefix = '/data/instruct-pix2pix/unzipped_images/'

    with wds.ShardWriter(
        f"{OUTPUT_DIR}/{pid:05d}_%06d.tar", maxcount=CHUNK_SIZE
    ) as sink:
        
        for dt in tqdm(sub_ds):
            # Each sample has single dict. Each dict has two keys (image_list, text).
            data_out = {}
            img_list = []
            try:
                # load image
                img_before = Image.open(
                os.path.join(image_root_prefix, dt['folder_name'], dt['img_before'])
                ).convert("RGB")
                buffered = BytesIO()
                img_before.save(buffered, format="JPEG")
                img_before_str = base64.b64encode(buffered.getvalue())
                
                img_after = Image.open(
                    os.path.join(image_root_prefix, dt['folder_name'], dt['img_after'])
                ).convert("RGB")
                buffered = BytesIO()
                img_after.save(buffered, format="JPEG")
                img_after_str = base64.b64encode(buffered.getvalue())
                    
                # convert to base64
                img_list.append(img_before_str.decode("utf-8"))
                img_list.append(img_after_str.decode("utf-8"))
                # 0th: before editing, 1th: after editing
                data_out["image"] = img_list
                
                # add text
                data_out["text"] = f"This is an image of {dt['input']}. Please {dt['edit']}."
            
                sink.write({
                    "__key__": dt['img_before'].split("_")[0],
                    "json": json.dumps(data_out),
                })
                n_samples += 1
            
            except Exception as e:
                print(f"Error processing: {e}")
                break
    
    return n_samples



def convert_all(ds, cores):

    ds = ds["train"]

    total_chunks = math.ceil(len(ds) / CHUNK_SIZE)
    args = []

    print(f"Total chunks = {total_chunks}")
    
    for chunk_id in tqdm(range(total_chunks)):
        st = chunk_id * CHUNK_SIZE
        ed = min((chunk_id + 1) * CHUNK_SIZE, len(ds))
        sub_ds = ds.select(list(range(st,ed)))
        args.append((sub_ds, chunk_id))

    n_procs = min(cores, total_chunks)

    print(f"Total processes = {n_procs}")
    print(f"Number of processes = {n_procs}")
    print("Start processing...")
    
    with Pool(processes=n_procs) as pool:
        r = pool.starmap(convert, args) # (sub_ds, chunk_id)
        
    print(f"Total examples = {sum(r)}")



if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    ds = datasets.load_dataset('json', data_files=DATA_ROOT)
    cores = os.cpu_count()

    convert_all(ds, cores)
    