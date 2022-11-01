import numpy as np
import pandas as pd
import faiss
from transformers import BertJapaneseTokenizer, BertModel
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    num_of_preds = 16
    device = "cuda:0"
    text="おバイオ"

    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval().to(device)
    
    vec_array = np.load("vec_array.npz", allow_pickle=True)['arr_0']
    df = pd.read_json(
        "exemplars.jsonl",
        orient="records",
        lines=True,
    )
    
    emb = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
    with torch.inference_mode():
        output=F.normalize(model(**emb.to(device))["pooler_output"].cpu())
    index = faiss.IndexFlatIP(vec_array.shape[1])
    index.add(vec_array)
    D, I = index.search(output.detach().numpy(), num_of_preds)
    preds=I[0][:5]
    for id, pred in enumerate(preds):
        print(df.to_dict("records")[pred]["text"])
        print(D[0][id])
if __name__ == "__main__":
    main()