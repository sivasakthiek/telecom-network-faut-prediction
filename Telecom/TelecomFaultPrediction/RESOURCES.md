# Telecom NOC Chatbot Architecture Resources

## Knowledge

- [Codebase: app.py](file:///d:/Siva1/Siva/Telecom/TelecomFaultPrediction/app.py)
  Streamlit entrypoint containing `process_chat_query`, session state coordination, and NOC Command Center tabs.
- [Codebase: llm_service.py](file:///d:/Siva1/Siva/Telecom/TelecomFaultPrediction/src/llm_service.py)
  The core routing and generation logic containing `route_query`, `generate_answer`, and the offline local fallback methods.
- [Codebase: rag_service.py](file:///d:/Siva1/Siva/Telecom/TelecomFaultPrediction/src/rag_service.py)
  FAISS-based vector database containing embedding encoders, nearest-neighbor searches, and fast lookup mappings.

## Wisdom (Communities)

- [Streamlit Community Forum](https://discuss.streamlit.io)
  Recommended for dashboard layout discussions and troubleshooting Streamlit session state behaviors.
- [OpenRouter Documentation](https://openrouter.ai/docs)
  Recommended for tracking API schemas, free tier limits (such as the 429 rate limit), and model configurations.
