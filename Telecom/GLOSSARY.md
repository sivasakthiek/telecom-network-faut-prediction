# Telecom Fault Prediction Glossary

This glossary contains the core vocabulary representing the hybrid NOC intelligence dashboard and RAG architecture.

## Terms

**RAG (Retrieval-Augmented Generation)**:
A technique where relevant telemetry summaries are retrieved from a FAISS vector database and injected into the LLM context to answer semantic user queries.
_Avoid_: Direct prompt lookup, vector search answering.

**SHAP (SHapley Additive exPlanations)**:
A game-theoretic framework used to calculate the contribution of each telemetry feature (such as log type volume or location risk) to a device's specific fault prediction.
_Avoid_: Anomaly detection scores, error features.

**What-If Engine**:
A deterministic, rule-based inference module that maps prediction probabilities and SHAP features to potential escalation scenarios if a device fault is left unresolved.
_Avoid_: Scenario LLM, prediction simulator.

**Query Router**:
A classifier in the LLM service that determines whether a user query is numerical (Pandas), semantic (RAG), or hypothetical (What-If), ensuring it is routed to the correct execution pipeline.
_Avoid_: Intent parser, search classifier.

**Health Score**:
A mathematical score ($100 - (P(Class 1)*40 + P(Class 2)*100)$) representing device status: Healthy ($\ge 70$), Warning ($40-69$), or Critical ($< 40$).
_Avoid_: Reliability index, priority score.
