# 🧠 CertBuddyAI – CFR RAG System

> A Retrieval-Augmented Generation (RAG) pipeline deployed on AWS to query CFR (Code of Federal Regulations) PDFs — including text, tables, and figures — using Amazon Bedrock and FAISS for vector retrieval.

## 📘 Overview

This project implements an end-to-end **RAG (Retrieval-Augmented Generation)** system capable of answering queries about U.S. Federal Motor Vehicle Safety Standards (FMVSS) regulations contained in multi-column CFR PDF documents.

It:
- Extracts **text, tables, and figures** from complex PDF layouts  
- Creates **vector embeddings** using **Amazon Bedrock’s `cohere.embed-v4` model**  
- Stores vectors in a **FAISS index** for semantic retrieval  
- Uses **Claude 4.5 Sonnet** for grounded text generation  
- Is deployed on **AWS EC2**, served via **FastAPI**  
- Has a **web frontend** hosted on **Amazon S3**  

## 🏗️ System Architecture (Mermaid Diagram)

```mermaid
flowchart TD
    subgraph User["🌐 User Browser"]
        F[Frontend (HTML/JS on S3)]
    end

    subgraph S3["🪣 Amazon S3 (Static Website)"]
        F -->|HTTP POST /ask| B
    end

    subgraph EC2["💻 EC2 Instance (Ubuntu + FastAPI)"]
        B[FastAPI RAG API<br>Uvicorn @8000]
        FAI[FAISS Vector Index]
        BED[Amazon Bedrock<br>(Claude 4.5 + Cohere Embed)]
        B -->|Retrieve + Embed| FAI
        B -->|Generate Answer| BED
    end

    subgraph LocalData["📄 Local Data"]
        PDF[Extracted CFR PDFs<br>(Text + Tables + Figures)]
        JSON[JSON Chunks<br>(text/table/figure)]
        FAISS[(FAISS Index<br>vectors.faiss)]
    end

    PDF -->|parse_pdfs.py| JSON -->|build_faiss.py| FAISS
    FAI --> B
    BED --> B
```

... (rest of markdown truncated for brevity) ...
