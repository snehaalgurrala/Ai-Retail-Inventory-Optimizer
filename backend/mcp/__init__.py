"""MCP layer for the chatbot.

Exposes a fixed, allow-listed catalog of read-only tools that answer strictly
from Oracle (the single source of truth) by reusing the existing repository and
analytics services. The LLM selects tools and explains results; it never writes
SQL and never sees the database directly.

No RAG, FAISS, embeddings, vector databases, or document retrieval are used on
this path.
"""
