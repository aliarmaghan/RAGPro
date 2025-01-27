
---

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/md-ali-armaghan/)&nbsp;
[![Twitter](https://img.shields.io/twitter/follow/armaghan78?label=Follow%20@armaghan78&style=social)](https://x.com/armaghan78)&nbsp;
[![Instagram](https://img.shields.io/badge/Instagram-Follow-E4405F)](https://www.instagram.com/be_armaghan?igsh=bjd2cDBtcW5mdTht)&nbsp;
[![Share on Twitter](https://img.shields.io/badge/Share-Twitter-1DA1F2)](https://twitter.com/intent/tweet?text=Explore%20RAGPro%20%E2%9A%99%EF%B8%8F%20https://github.com/aliarmaghan/RAGPro)&nbsp;
[![Star on GitHub](https://img.shields.io/github/stars/aliarmaghan/RAGPro?style=social)](https://github.com/aliarmaghan/RAGPro/stargazers)

> If you find this repository helpful, please consider giving it a star⭐️

# RAGPro 🚀
![RAGPro Banner](https://via.placeholder.com/1920x400.png?text=RAGPro+Retrieval-Augmented+Generation) <!-- Add your banner URL -->

Welcome to **RAGPro**, a comprehensive repository dedicated to mastering Retrieval-Augmented Generation (RAG) systems. From foundational concepts to advanced implementations, this repo provides curated resources, code examples, and practical guides to help you excel in building intelligent, knowledge-enhanced AI systems.

## 📚 Table of Contents
- [Features](#-key-features)
- [Advanced RAG Techniques](#-advanced-rag-techniques)
- [Agentic RAG Techniques](#-agentic-rag-techniques)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🚀 Key Features
- **Comprehensive Coverage**: From basic RAG concepts to advanced and agentic techniques.
- **Hands-On Notebooks**: Colab-ready examples for quick experimentation.
- **Cutting-Edge Tools**: Implementations using LangChain, Hugging Face, Pinecone, ChromaDB, and more.
- **Real-World Applications**: Practical use cases for question-answering, document retrieval, and conversational AI.
- **Community-Driven**: Designed for ease of contribution and collaboration.

---

<!-- ## Advanced RAG Techniques⚙️
Here are the details of all the Advanced RAG techniques covered in this repository.

| Technique                    | Tools                        | Description                                                       | Notebooks |
|------------------------------|------------------------------|-------------------------------------------------------------------|-----------|
| Naive RAG                    | LangChain, Pinecone, Athina AI | Combines retrieved data with LLMs for simple and effective responses. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/naive_rag.ipynb) |
| Hybrid RAG                   | LangChain, Chromadb, Athina AI | Combines vector search and traditional methods like BM25 for better information retrieval. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/hybrid_rag.ipynb) |
| Hyde RAG                     | LangChain, Weaviate, Athina AI | Creates hypothetical document embeddings to find relevant information for a query. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/hyde_rag.ipynb) |
| Parent Document Retriever    | LangChain, Chromadb, Athina AI | Breaks large documents into small parts and retrieves the full document if a part matches the query. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/parent_document_retriever.ipynb) |
| RAG Fusion                   | LangChain, LangSmith, Qdrant, Athina AI | Generates sub-queries, ranks documents with Reciprocal Rank Fusion, and uses top results for accurate responses. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/fusion_rag.ipynb) |
| Contextual RAG               | LangChain, Chromadb, Athina AI | Compresses retrieved documents to keep only relevant details for concise and accurate responses. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/contextual_rag.ipynb) |
| Rewrite Retrieve Read        | LangChain, Chromadb, Athina AI | Improves query, retrieves better data, and generates accurate answers. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/rewrite_retrieve_read.ipynb) |
| Unstructured RAG             | LangChain, LangGraph, FAISS, Athina AI, Unstructured | Designed to handle documents that combine text, tables, and images. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/basic_unstructured_rag.ipynb) |

---

## Agentic RAG Techniques⚙️
Here are the details of all the Agentic RAG techniques covered in this repository.

| Technique                    | Tools                        | Description                                                       | Notebooks |
|------------------------------|------------------------------|-------------------------------------------------------------------|-----------|
| Basic Agentic RAG            | LangChain, FAISS, Athina AI  | Uses AI agents to find and generate answers using tools like vectordb and web searches. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/agentic_rag_techniques/basic_agentic_rag.ipynb) |
| Corrective RAG               | LangChain, LangGraph, Chromadb, Athina AI | Refines relevant documents, removes irrelevant ones, or performs web searches. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/agentic_rag_techniques/corrective_rag.ipynb) |
| Self RAG                     | LangChain, LangGraph, FAISS, Athina AI | Reflects on retrieved data to ensure accurate and complete responses. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/agentic_rag_techniques/self_rag.ipynb) |
| Adaptive RAG                 | LangChain, LangGraph, FAISS, Athina AI | Adjusts retrieval methods based on query type, using indexed data or web search. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/agentic_rag_techniques/adaptive_rag.ipynb) |
| ReAct RAG                    | LangChain, LangGraph, FAISS, Athina AI | Combines reasoning and retrieval for context-aware responses. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/athina-ai/rag-cookbooks/blob/main/agentic_rag_techniques/react_rag.ipynb) | -->

---

## ⚡ Installation
```bash
# Clone the repository
git clone https://github.com/aliarmaghan/RAGPro.git
cd RAGPro

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export GROQ_API_KEY="your-groq-api-key"
```

## 💻 Usage
```python
from langchain.chains import RetrievalQA

# Initialize your RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Execute query
response = qa_chain.run("What is the difference between AI and ML?")
print(response)
```

## 🛠️ Tech Stack
- **Core AI**: OpenAI GPT, LangChain, Hugging Face
- **Vector Databases**: Pinecone, ChromaDB, FAISS
- **APIs**: Groq API, CrewAI
- **Frameworks**: FastAPI, Streamlit
- **Deployment**: Docker, Streamlit Cloud

## 🤝 Contributing
We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/amazing-feature`.
3. Commit your changes: `git commit -m 'Add amazing feature'`.
4. Push to the branch: `git push origin feature/amazing-feature`.
5. Open a pull request.

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📧 Contact
**Md Ali Armaghan**  
[![Email](https://img.shields.io/badge/Email-aliarmaghan78@gmail.com-blue?logo=gmail)](mailto:aliarmaghan@example.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/md-ali-armaghan/)
