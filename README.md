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

## System Architecture

```mermaid
flowchart TD
    A[User Browser] -->|HTTP POST /ask| B[Amazon S3 (Static Website)]

    B --> C[EC2 Instance (FastAPI Backend)]
    C --> D[FAISS Vector Index]
    C --> E[Amazon Bedrock (Claude 4.5 + Cohere Embed)]

    F[Extracted PDFs (Text + Tables + Figures)] --> G[JSON Chunks]
    G --> D

    C -->|Retrieve Context| D
    C -->|Generate Answer| E
```
