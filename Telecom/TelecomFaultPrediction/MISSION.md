# Mission: Telecom NOC Chatbot Architecture

## Why
The user wants to understand the exact end-to-end architecture flow of the dashboard's AI network assistant (RAG, routing, database lookup, pandas execution, and LLM synthesis) so they can successfully maintain, debug, and scale the NOC chatbot interface.

## Success looks like
- Can trace a user query from the Streamlit text input to the final generated or fallback response.
- Understands the routing decision logic between numerical query execution (Pandas) and semantic searches (FAISS + LLM).
- Comprehends how context (such as active inspected device status) is injected into the LLM system prompt.
- Knows how the system behaves under rate-limiting (429) or offline conditions.

## Constraints
- Deliverables must be formatted as beautiful, easy-to-read reference materials (HTML files).
- Keep code explanations tightly bound to the actual implementation in `app.py`, `src/llm_service.py`, and `src/rag_service.py`.

## Out of scope
- Training or fine-tuning the underlying LLM models.
- Deep dive into how the core XGBoost machine learning model was trained (unless related to predictions showing up in the chatbot's device context).
