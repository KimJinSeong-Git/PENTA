# core/config.py
import os

# Directory paths
EVAL_DIR = "./evaluation_results"
DECOMP_DIR = "./decomposition_results"

# Directory initialization
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(DECOMP_DIR, exist_ok=True)

# Default supported models and pricing (USD per 1M tokens)
DEFAULT_MODELS = {
    "google/gemini-2.5-pro": {"input_cost_1m": 1.25, "output_cost_1m": 10.00},
    "google/gemini-2.5-flash": {"input_cost_1m": 0.30, "output_cost_1m": 2.50},
    "google/gemini-2.5-flash-lite": {"input_cost_1m": 0.10, "output_cost_1m": 0.40},
    "google/gemma-4-31b-it": {"input_cost_1m": 0.12, "output_cost_1m": 0.35},
    "anthropic/claude-4.8-opus": {"input_cost_1m": 5.00, "output_cost_1m": 25.00},
    "anthropic/claude-5-sonnet": {"input_cost_1m": 2.00, "output_cost_1m": 10.00},
    "anthropic/claude-4.5-haiku": {"input_cost_1m": 1.00, "output_cost_1m": 5.00},
    "openai/gpt-5.6-sol-pro": {"input_cost_1m": 5.00, "output_cost_1m": 30.00},
    "openai/gpt-5.6-luna": {"input_cost_1m": 1.00, "output_cost_1m": 6.00},
    "openai/gpt-5.4-nano": {"input_cost_1m": 0.20, "output_cost_1m": 1.25}
}