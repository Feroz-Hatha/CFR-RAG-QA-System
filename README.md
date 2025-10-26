# RAG Based QA System For CFR Documents

> A Retrieval-Augmented Generation (RAG) pipeline deployed on AWS to query CFR (Code of Federal Regulations) PDFs — including text, tables, and figures — using Amazon Bedrock and FAISS for vector retrieval.

## Overview

This project implements an end-to-end **RAG** system capable of answering queries about U.S. Federal Motor Vehicle Safety Standards (FMVSS) regulations contained in multi-column CFR PDF documents.

It:
- Extracts **text, tables, and figures** from complex PDF layouts  
- Creates **vector embeddings** using **AWS Bedrock’s `Amazon Titan Text Embeddings v2` model**  
- Stores vectors in a **FAISS index** for semantic retrieval  
- Uses **Claude Sonnet 4.5** for grounded text generation  
- Is deployed on **AWS EC2**, served via **FastAPI**  
- Has a **web frontend** hosted on **AWS S3**  

## System Architecture (Mermaid Diagram)

```mermaid
flowchart TD
    subgraph User["User Browser"]
        F[Frontend (HTML/JS on S3)]
    end

    subgraph S3["Amazon S3 (Static Website)"]
        F -->|HTTP POST /ask| B
    end

    subgraph EC2["EC2 Instance (FastAPI Backend)"]
        B[FastAPI RAG API (Uvicorn :8000)]
        FAI[FAISS Vector Index]
        BED[Amazon Bedrock (Claude Sonnet 4.5 + Amazon Titan Text Embeddings v2)]
        B -->|Retrieve + Embed| FAI
        B -->|Generate Answer| BED
    end

    subgraph LocalData["Local Data (Extracted PDFs)"]
        PDF[Extracted CFR PDFs (Text + Tables + Figures)]
        JSON[JSON Chunks (text/table/figure)]
        FAISS[(FAISS Index vectors.faiss)]
    end

    PDF -->|parse_pdfs.py| JSON -->|build_faiss.py| FAISS
    FAI --> B
    BED --> B
```
