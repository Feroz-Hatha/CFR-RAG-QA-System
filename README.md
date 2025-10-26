# RAG Based QA System For CFR Documents

> A Retrieval-Augmented Generation (RAG) pipeline deployed on AWS to query CFR (Code of Federal Regulations) PDFs — including text, tables, and figures — using Amazon Bedrock and FAISS for vector retrieval.

## Overview

This project implements an end-to-end **RAG** system capable of answering queries about U.S. Federal Motor Vehicle Safety Standards (FMVSS) regulations contained in multi-column CFR PDF documents. The deployed demo can be found [here](http://certbuddy-rag-ui.s3-website.us-east-2.amazonaws.com/).

It:
- Extracts **text, tables, and figures** from complex PDF layouts  
- Creates **vector embeddings** using **AWS Bedrock’s `Amazon Titan Text Embeddings v2` model**  
- Stores vectors in a **FAISS index** for semantic retrieval  
- Uses **Claude Sonnet 4.5** for grounded text generation  
- Is deployed on **AWS EC2**, served via **FastAPI**  
- Has a **web frontend** hosted on **AWS S3**  
