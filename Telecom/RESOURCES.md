# Telecom Fault Prediction & Hybrid RAG Resources

## Knowledge

- [Article: "SHAP Explainers" — SHAP Official Documentation](https://shap.readthedocs.io/)
  Official docs on TreeExplainer, calculation of Shapley values, and feature contributions. Use for: understanding how SHAP features are calculated in `src/predictor.py`.
- [Article: "Retrieval-Augmented Generation for LLMs" — LlamaIndex / LangChain Guides](https://docs.llamaindex.ai/)
  Guides on naive, conversational, and hybrid RAG retrieval architectures. Use for: understanding vector search and context grounding in `src/rag_service.py`.
- [Article: "XGBoost Documentation" — XGBoost Developers](https://xgboost.readthedocs.io/)
  Official documentation for the XGBoost gradient boosting library. Use for: understanding how the pre-trained classifier works in `src/predictor.py`.
- [Article: "Streamlit Custom Components" — Streamlit Docs](https://docs.streamlit.io/library/components)
  Guidelines for rendering custom iframe HTML5/JS widgets inside Streamlit. Use for: understanding `components.html` used in `app.py` for the network topology canvas.

## Wisdom (Communities)

- [r/MachineLearning](https://reddit.com/r/MachineLearning)
  Subreddit dedicated to machine learning research and practices. Use for: model explainability discussions.
- [r/Streamlit](https://reddit.com/r/streamlit)
  Community for dashboard styling and custom rendering tricks. Use for: dashboard component performance troubleshooting.
