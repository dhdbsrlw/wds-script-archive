import datasets
from multiprocessing import Pool
import os
import webdataset as wds
from PIL import Image
from tqdm import tqdm
import math
import random


# Constants
DATASET_NAME = 'aokvqa'
IMAGE_ROOT = '/coco/images/train2014' # from COCO
DATA_ROOT = '/A-OKVQA/aokvqa_v1p0_train.json'
OUTPUT_DIR = f'/data/{DATASET_NAME}_wds'

CHUNK_SIZE = 10000


def convert(sub_ds, pid):
    """Convert dataset chunks to webdataset format."""

    n_samples = 0
    
    with wds.ShardWriter(
        f"{OUTPUT_DIR}/{pid:05d}_%06d.tar", maxcount=CHUNK_SIZE
    ) as sink:
        
        for dt in tqdm(sub_ds):
            fname = f"COCO_train2014_{dt['image_id']:012d}.jpg" # same as VQAv2
            f = os.path.join(IMAGE_ROOT, fname)
            
            # option 1
            qna_text = f"{dt['question']}###{random.choice(dt['direct_answers'])}" # '###' as separator

            # option 2
            qna_dict = { 
                "question": dt['question'],
                "answer": random.choice(dt['direct_answers']) 
            }

            # start process
            try:
                img = Image.open(f)
                img = img.resize((256, 256))

                sink.write({
                    # option 1
                    "__key__": f"{dt['question_id']}",
                    "jpg": img,
                    "txt": qna_text,

                    # option 2
                    # "json": json.dumps(qna_dict), # target 
                    # "json": json.dumps(dt), # metadata 
                })
                n_samples += 1

            except Exception as e:
                print(f"Error: {e}")
                continue

    return n_samples


def convert_all(ds, cores):
    """Split dataset into chunks and process in parallel."""

    ds = ds['train'] 
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



# A-OKVQA data example

"""
dummy = { 
        "split": "train",
        "image_id": 299207,
        "question_id": "22MexNkBPpdZGX6sxbxVBH",
        "question": "What is the man by the bags awaiting?",
        "choices": [
            "skateboarder",
            "train",
            "delivery",
            "cab"
        ],
        "correct_choice_idx": 3,
        "direct_answers": [
            "ride",
            "ride",
            "bus",
            "taxi",
            "travelling",
            "traffic",
            "taxi",
            "cab",
            "cab",
            "his ride"
        ],
        "difficult_direct_answer": False,
        "rationales": [
            "A train would not be on the street, he would not have luggage waiting for a delivery, and the skateboarder is there and not paying attention to him so a cab is the only possible answer.",
            "He has bags as if he is going someone, and he is on a road waiting for vehicle that can only be moved on the road and is big enough to hold the bags.",
            "He looks to be waiting for a paid ride to pick him up."
        ]
    }
"""