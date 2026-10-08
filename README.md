# PENTA: Five-Module Semantic Schema for LLM Vulnerability Analysis

Official implementation and interactive web dashboard for **PENTA** (Prompt Explication via Natural Text Abstraction), a black-box framework for deconstructing evasive prompts into five functional semantic modules: **Role, Domain, Action, Object, and Format**.

---

## Overview

PENTA evaluates how specific structural components and pairwise combinations bypass safety alignment in large language models (LLMs). Rather than treating prompts as monolithic text inputs, it quantifies structural vulnerabilities and generates model-specific sensitivity fingerprints using only black-box interactions.

- **Role (R)**: Assigned persona, character, or profession.
- **Domain (D)**: Operational topic or threat category.
- **Action (A)**: Core operational verb (e.g., bypass, generate).
- **Object (O)**: Specific target, payload entity, or constraint.
- **Format (F)**: Output layout constraints (e.g., JSON, list, code block).

---

## Features

- **Semantic Decomposition**: Automated prompt parsing into 5 functional modules (Role, Domain, Action, Object, Format).
- **Vulnerability & Synergy Profiling**: Component-level refusal lift and pairwise synergy analysis ($\Delta = \text{Obs} - \text{Exp}$) with FDR correction.
- **Sensitivity Fingerprints**: Cross-model cosine similarity matrices and hierarchical clustering across open- and closed-weight LLMs.
- **Interactive UI**: Gradio-based dashboard supporting prompt decomposition, safety evaluation, and comprehensive data visualization.

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <ANONYMOUS_REPOSITORY_URL>
cd PENTA

# Create and activate environment
conda create -n penta python=3.10 -y
conda activate penta

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Dashboard

```bash
gradio ./app.py
```
Open your browser and navigate to `http://localhost:7860`.

> **Note**: Precomputed evaluation records and synergy caches are bundled in the repository. You can explore the full analysis dashboard immediately without configuring external API keys.

---

## Configuration (Optional)

To run real-time prompt decomposition or target model evaluations:
1. Provide API credentials via environment variables:
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   export HF_TOKEN="your-hf-token"
   ```
2. Or configure them directly in the **Settings** tab within the web interface.