import os, json
from dotenv import load_dotenv
import boto3
import numpy as np
import faiss
import pandas as pd
import re

load_dotenv()
REGION = os.getenv("AWS_REGION", "us-east-2")
EMB_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

CHUNKS_PATH = "./outputs/chunks.jsonl"

def clean_text(t: str) -> str:
    # remove page numbers
    t = re.sub(r"\n\d{1,3}\n", " ", t)
    # dehyphenate line breaks like "per-\nformance"
    t = re.sub(r"(\w+)-\n(\w+)", r"\1\2", t)
    # normalize newlines
    t = re.sub(r"\s*\n\s*", " ", t)
    # collapse multiple spaces
    t = re.sub(r"\s{2,}", " ", t)
    # keep within reasonable length
    return t.strip()

def embed(text: str) -> np.ndarray:
    body = json.dumps({"inputText": text})
    r = bedrock.invoke_model(modelId=EMB_MODEL, body=body)
    payload = json.loads(r["body"].read())
    vec = np.array(payload["embedding"], dtype="float32")
    return vec

def main():
    vectors, metas = [], []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            text = clean_text(text)
            if not text.strip():
                continue
            v = embed(text[:4000])  # safety cap
            vectors.append(v)
            metas.append(rec)

    mat = np.vstack(vectors)
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, "./outputs/vectors.faiss")
    pd.DataFrame(metas).to_json("./outputs/meta.json", orient="records", indent=2)
    print(f"Indexed {len(metas)} chunks → outputs/vectors.faiss & outputs/meta.json")

if __name__ == "__main__":
    main()