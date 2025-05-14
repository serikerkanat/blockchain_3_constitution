# AI Constitutional Assistant for Kazakhstan

![Demo Screenshot](./demo.png)

An AI-powered assistant that answers questions about the Constitution of Kazakhstan using Retrieval-Augmented Generation (RAG) technology.

## Features

- 💬 Chat interface for constitutional queries
- 📁 Document upload (PDF/DOCX/TXT) with context-aware answers
- 🧠 Local AI processing via Ollama (Phi-3 3.8B model)
- 📊 Vector database (ChromaDB) for efficient retrieval
- 🌐 Optional web-based constitution loading

## Installation

1. **Clone repository**:
   ```bash
   git clone https://github.com/serikerkanat/blockchain_3_constitution.git
   cd blockchain_3_constitution
   ```
Set up environment:
  ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
```
Install dependencies:
```bash
    pip install -r requirements.txt
```
Download AI model:
```bash
    ollama pull phi3:3.8b
```
Usage
```bash
streamlit run app.py
```
The app will open in your browser at http://localhost:8501

#How to Use:
Upload constitutional documents (or use default)

Ask questions in the chat interface

Receive answers with source references

#Technical Stack
Framework: Streamlit

AI Model: Phi-3 3.8B (via Ollama)

Vector DB: ChromaDB

Embeddings: OllamaEmbeddings

Text Processing: LangChain

Screenshots:
![Снимок экрана 2025-05-14 134252](https://github.com/user-attachments/assets/cb5658ec-1885-4733-b051-013cd09324a2)
![Снимок экрана 2025-05-14 135615](https://github.com/user-attachments/assets/ae3f6f24-cb0c-4830-9aec-eb87ed234277)
![Снимок экрана 2025-05-14 135722](https://github.com/user-attachments/assets/669fbd3e-e7aa-45fa-b44c-0568b6ae6afe)
![Снимок экрана 2025-05-14 135727](https://github.com/user-attachments/assets/f2c7189a-11a1-4e18-bf4b-fbea020eb8e7)
![Снимок экрана 2025-05-14 142420](https://github.com/user-attachments/assets/44f5df6e-1296-4e03-ac76-21d9ec4b49df)




