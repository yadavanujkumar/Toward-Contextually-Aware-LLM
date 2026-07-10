# Toward Contextually Aware LLMs: Affective Safety & Memory Prototype

This repository contains the official reference prototype for an advanced agentic architecture designed to provide context-aware, empathetic, and structurally safe LLM interactions. It implements a 4-layer architecture combining multi-objective retrieval, real-time token compression, latent emotional trajectory tracking, and strict boundary guardrails.

## Features & Architecture

This prototype demonstrates a complete end-to-end pipeline consisting of:

1. **ByteRover Context Tree (Layer 1)**: An agent-native, atomic filesystem that securely stores and retrieves interaction logs and behavioral protocols as memory vectors.
2. **EI-CUS Retrieval Engine (Layer 1)**: Emotion-Intent Contextual Utility Scoring. Uses 0-1 Knapsack algorithms to fetch the highest density utility context (combining Semantic, Affective, Intent, and Temporal matching) under strict token budgets.
3. **MICA/MAPO Latent Empathy Tracker**: Maps user distress into a 3D physical coordinate space (Cognitive, Affective, Proactive) and mathematically calculates the Intrinsic Distress Resolution (IDR) reward per turn.
4. **Token Efficiency Simulator (Layer 4)**: Evaluates the performance gains of Evaluator Head-based Prompt Compression (EHPC) and RocketKV sparse cache eviction mechanics to minimize VRAM usage.
5. **Affective Safety Guardrail (Layer 3)**: Uses AHaPairs DPO boundary checks to detect and surgically intercept "affective hallucinations" (e.g., an AI claiming it is human or has feelings).
6. **Agent Memory Guard**: Scans incoming interactions for malicious payloads (e.g., "drop database") and intercepts them before they can poison the long-term context database.
7. **Gemini Live LLM Integration**: Dynamically synthesizes the compressed context into natural language responses via Google's `gemini-1.5-flash` model.

## Installation

This prototype relies on local NLP kernels for mathematical embeddings and context sizing.

### 1. Install Dependencies
```bash
pip install sentence-transformers tiktoken matplotlib tf-keras google-generativeai
```

### 2. Configure Gemini API
The system uses the Gemini API to power the natural language generation module. You must expose your API key to your environment variables:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```
**Mac/Linux:**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

*Note: If no API key is detected, the script will safely fallback to simulating responses so the pipeline won't crash.*

## Usage

Simply run the main entry point to launch the interactive sandbox evaluation suite:

```bash
python main.py
```

### What to expect during execution:
The sandbox runs a multi-turn evaluation suite testing complex stress scenarios (such as mixed emotional distress paired with malicious code-injection attempts). 

For each turn, the CLI will output:
- **System Diagnostics**: Detailed mathematical metrics (Context Weights, Token Compression Rates, Latent Energy drops, and VRAM saved).
- **Plain-English Explanations**: A human-readable breakdown translating the metrics to explain exactly *why* the AI acted the way it did, what memories it pulled, and how it protected itself.
- **Agent Response**: The final synthesized output from the LLM.

At the end of execution, a 3D visualization of the user's emotional trajectory is saved locally as `trajectory_plot.png`.
