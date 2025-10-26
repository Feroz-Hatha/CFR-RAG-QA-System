import os, json
from dotenv import load_dotenv
import numpy as np
import faiss
import boto3

load_dotenv()
REGION = os.getenv("AWS_REGION", "us-east-2")
GEN_MODEL = os.getenv("BEDROCK_GEN_MODEL", "anthropic.claude-sonnet-4-5-20250929-v1:0")
EMB_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

INDEX_PATH = "./outputs/vectors.faiss"
META_PATH = "./outputs/meta.json"

def embed(text: str) -> np.ndarray:
    body = json.dumps({"inputText": text})
    r = bedrock.invoke_model(modelId=EMB_MODEL, body=body)
    payload = json.loads(r["body"].read())
    return np.array(payload["embedding"], dtype="float32")

def search(query: str, k=8):
    qv = embed(query)
    faiss.normalize_L2(qv.reshape(1,-1))
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    D, I = index.search(qv.reshape(1,-1), k)
    results = [meta[i] for i in I[0]]
    return results

def answer(query: str, contexts):
    ctx = ""
    for c in contexts:
        tag = c.get("kind","text").upper()
        ctx += f"- [{tag}] {c['doc_id']} p.{c['page']}: {c['text']}\n"

    prompt = (
        "You are a strict RAG assistant. Answer ONLY from CONTEXT.\n"
        "Cite like (doc_id p.page). If unsure, say you don't know.\n\n"
        f"QUESTION:\n{query}\n\nCONTEXT:\n{ctx}"
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        "max_tokens": 600,
        "temperature": 0.0
    })

    r = bedrock.invoke_model(modelId=GEN_MODEL, body=body)
    payload = json.loads(r["body"].read())

    # Claude 4 / 4.5 format
    out_text = ""
    if "output" in payload and "message" in payload["output"]:
        for blk in payload["output"]["message"].get("content", []):
            if blk.get("type") == "text":
                out_text += blk.get("text", "")
    else:
        out_text = json.dumps(payload, indent=2)

    return out_text

def cli():
    print("Ask a question (Ctrl+C to exit):")
    while True:
        q = input("> ").strip()
        if not q: 
            continue
        ctx = search(q, k=8)
        ans = answer(q, ctx)
        print("\n=== ANSWER ===\n")
        print(ans)
        print("\n=== TOP CONTEXTS ===\n")
        for i, c in enumerate(ctx, 1):
            print(f"{i}. [{c['kind']}] {c['doc_id']} p.{c['page']}")
        print()

if __name__ == "__main__":
    cli()
