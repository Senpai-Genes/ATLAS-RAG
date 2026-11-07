# ATLAS — Advanced PDF RAG Assistant

Atlas is a local-first, full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents.

Atlas combines RAPTOR hierarchical indexing, ColBERT retrieval, intelligent query processing, persistent conversation memory, and the Qwen3 language model to generate context-aware answers with source citations.

The application was developed using Python, FastAPI, React, and Ollama.

The Python backend was developed and tested in Ubuntu Linux using Windows Subsystem for Linux (WSL2), while the React frontend was developed using Windows PowerShell.

## Features

- PDF uploading and document management
- RAPTOR hierarchical document indexing
- ColBERT late-interaction retrieval
- Intelligent query routing and decomposition
- Context-aware follow-up questions
- Persistent conversation history using SQLite
- Local answer generation using Qwen3 through Ollama
- Source citations and page references
- Background PDF indexing
- React-based user interface
- Document deletion and collection management

## How Atlas Works

Atlas operates through two major processes: document indexing and question answering.

### 1. Document Indexing

When a user uploads a PDF, Atlas extracts its text and divides it into smaller chunks.

RAPTOR creates a hierarchical representation of the document using SentenceTransformers, UMAP, Gaussian Mixture Models, and language-model-generated summaries.

This helps represent information at different levels of abstraction.

ColBERT then creates a retrieval index using token-level embeddings and late-interaction scoring.

The document, indexes, and metadata are stored locally for future retrieval.

### 2. Question Answering

When a user asks a question, Atlas follows these steps:

1. Retrieves previous conversation context when available.
2. Rewrites follow-up questions into standalone queries.
3. Analyzes the question using query routing and decomposition.
4. Retrieves relevant passages using ColBERT.
5. Builds a structured context with source references.
6. Sends the question and context to Qwen3 through Ollama.
7. Returns an answer with supporting citations.

## Technology Stack

Backend: Python 3.12, FastAPI, Uvicorn

Frontend: React, Vite, JavaScript

Language Model: Qwen3:4b

LLM Runtime: Ollama

Hierarchical Indexing: RAPTOR

Retrieval: ColBERT and RAGatouille

Embeddings: SentenceTransformers

Dimensionality Reduction: UMAP

Clustering: Gaussian Mixture Models

Database: SQLite

Development Environment: Ubuntu WSL2 and Windows

Version Control: Git and GitHub

---

# Installation Guide

Atlas was developed using Ubuntu Linux through Windows Subsystem for Linux (WSL2).

Windows users are recommended to run the Python backend inside Ubuntu WSL2 because ColBERT, PyTorch, and related dependencies may encounter compatibility issues on native Windows.

The React frontend can run in either Ubuntu or Windows Power
