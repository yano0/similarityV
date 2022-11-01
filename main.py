from transformers import BertJapaneseTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from typing import Dict, List, Union
from tqdm import tqdm
import torch.nn.functional as F
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_path", type=str, default="tweet.jsonl")
    parser.add_argument("--model", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    return parser.parse_args()


def get_data(path):
    with open(path) as f:
        text = f.readlines()
        part_text = text[:10000]
    return part_text

@torch.inference_mode()
def main():
    args = parse_args()
    device = args.device
    model_name = args.model
    df = pd.read_json(
    args.input_path,
    orient="records",
    lines=True,
    )
    tweets=[]
    for df_dict in df.to_dict("records"):
        tweets+=df_dict["tweet"]
    part_text = tweets
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval().to(device)
    
    def collate_fn(batch: List[str]) -> BatchEncoding:
        return tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
    
    dl = DataLoader(part_text, collate_fn=collate_fn, batch_size=16, num_workers=2)
    vec_list=[]
    for batch in tqdm(dl):
        output = model(**batch.to(device))
        vec_list.append(output["pooler_output"].cpu())
    df_ex=pd.DataFrame({"text":part_text})
    df_ex.to_json( "exemplars.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
        default_handler=str
    )
    vec_array = torch.cat(vec_list, dim=0)
    vec_array = F.normalize(vec_array)
    vec_array.detach().numpy()
    np.savez_compressed("./vec_array", vec_array)
if __name__ == "__main__":
    main()