from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, json, boto3, faiss, numpy as np
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv("AWS_REGION","us-east-2")
GEN_MODEL = os.getenv("BEDROCK_GEN_MODEL")
EMB_MODEL = os.getenv("BEDROCK_EMBED_MODEL")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_VECS = os.getenv("S3_VECS","vectors.faiss")
S3_META = os.getenv("S3_META","meta.json")

s3 = boto3.client("s3", region_name=REGION)
br = boto3.client("bedrock-runtime", region_name=REGION)

# Download FAISS data to local
os.makedirs("data", exist_ok=True)
s3.download_file(S3_BUCKET, S3_VECS, "data/vectors.faiss")
s3.download_file(S3_BUCKET, S3_META, "data/meta.json")

index = faiss.read_index("data/vectors.faiss")
with open("data/meta.json") as f:
    META = json.load(f)

def embed(text: str) -> np.ndarray:
    body = json.dumps({"inputText": text})
    resp = br.invoke_model(modelId=EMB_MODEL, body=body)
    payload = json.loads(resp["body"].read())
    vec = np.array(payload["embedding"], dtype="float32")
    return vec

def search(query: str, k=8):
    qv = embed(query)
    faiss.normalize_L2(qv.reshape(1,-1))
    D, I = index.search(qv.reshape(1,-1), k)
    return [META[i] for i in I[0]], (D[0].tolist(), I[0].tolist())

def build_prompt(query, contexts):
    lines = []
    for c in contexts:
        tag = c.get("kind","text").upper()
        lines.append(f"- [{tag}] {c['doc_id']} p.{c['page']}: {c['text']}")
    ctx = "\n".join(lines)
    prompt = (
        "You are a strict RAG assistant. Answer ONLY from CONTEXT.\n"
        "Cite like (doc_id p.page). If unsure, say you don't know.\n\n"
        f"QUESTION:\n{query}\n\nCONTEXT:\n{ctx}"
    )
    return prompt

def generate_answer_claude(prompt: str) -> str:
    # Claude 4/4.5 require anthropic_version; Claude 3.5 ignores extra fields
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role":"user","content":[{"type":"text","text":prompt}]}
        ],
        "max_tokens": 600,
        "temperature": 0.0
    })
    r = br.invoke_model(modelId=GEN_MODEL, body=body)
    payload = json.loads(r["body"].read())
    # Claude 4/4.5 format
    if "output" in payload and "message" in payload["output"]:
        chunks = payload["output"]["message"].get("content", [])
        return "".join(x.get("text","") for x in chunks if x.get("type")=="text")
    # Claude 3.5 format
    if "content" in payload:
        return "".join(x.get("text","") for x in payload["content"] if x.get("type")=="text")
    return json.dumps(payload)

class AskIn(BaseModel):
    query: str
    k: int = 8

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(body: AskIn):
    q = body.query.strip()
    if not q:
        raise HTTPException(400, "query required")
    ctx, _ = search(q, k=body.k)
    prompt = build_prompt(q, ctx)
    answer = generate_answer_claude(prompt)
    return {"answer": answer, "contexts": ctx}