# Telecom Fault Prediction Context

This context defines the core domain concepts for the Telecom Network Intelligence dashboard, which predicts network fault severity and provides a RAG-based AI assistant for querying network status.

## Language

**Location**:
A specific node or terminal in the telecommunications network where log events are monitored.
_Avoid_: Area, region, zone

**Fault Severity**:
The classification of network disruption severity at a location (0: No fault, 1: Minor fault, 2: Major fault).
_Avoid_: Error level, disruption grade

**Resource Type**:
A specific hardware or software component category within a network node that triggered the event.
_Avoid_: System component, hardware type

**Event Type**:
A categorized network signal or log category registered at a location.
_Avoid_: Signal type, alert category

**Log Feature**:
A diagnostic feature extracted from log entries at a location, paired with a volume count.
_Avoid_: Log entry, error code

**Health Score**:
A computed status indicator representing the operational health of a location based on severity and feature logs.
_Avoid_: Status score, performance metric

**RAG Assistant**:
An intelligent agent that answers natural-language questions using retrieved location data from a vector database and an LLM.
_Avoid_: General chatbot, search bot

**LLM Router**:
An intent classifier that decides whether a query should be resolved via structured database operations or semantic vector search.
_Avoid_: Command parser, classification model
