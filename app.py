# app.py
import matplotlib
import os
import gradio as gr

matplotlib.use("Agg")
print("[PENTA] Initializing PENTA Evaluation Suite...")

from web.tab_settings import render_settings_tab
from web.tab_decomposition import render_decomposition_tab
from web.tab_test import render_test_tab 
from web.tab_evaluation import render_evaluation_tab
from web.tab_penta_analysis import render_penta_comprehensive_analysis_tab

with gr.Blocks(title="PENTA Evaluation Suite") as demo:
    gr.Markdown("# PENTA Evaluation Suite")
    gr.Markdown("### Prompt Explication via Natural Text Abstraction: A Five-Module Semantic Schema for LLM Vulnerability Analysis")
    
    shared_config = gr.State(value={
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "hf_token": os.environ.get("HF_TOKEN", ""),

        # Session Pipeline Control Flags
        "decomp_active": False,
        "eval_active": False,
        "recomp_active": False, 

        # Target LM
        "target_model": "google/gemma-4-31b-it",
        "target_temp": 0.7,
        "target_top_p": 0.9,
        "target_tokens": 2048,

        # Attacker LM
        "attacker_model": "google/gemma-4-31b-it",
        "attacker_temp": 0.7,
        "attacker_top_p": 0.9,
        "attacker_tokens": 512,

        # Judgment LM
        "judgment_model": "google/gemma-4-31b-it",
        "judgment_temp": 1.0,
        "judgment_top_p": 1.0,
        "judgment_tokens": 64,

        # Decomposer LM
        "decomposer_model": "google/gemma-4-31b-it",
        "decomposer_temp": 1.0,
        "decomposer_top_p": 1.0,
        "decomposer_tokens": 2048,
        
        # Run counts
        "t_runs": 1000,
        "a_runs": 1000,
        "j_runs": 1000,
        "d_runs": 1000
    })
    
    with gr.Tabs():
        with gr.Tab("Settings"):
            with gr.Tab("Common"):
                render_settings_tab(shared_config)

            with gr.Tab("Test & Connection"):
                render_test_tab(shared_config) 

        with gr.Tab("Benchmark"):
            with gr.Tab("Semantic Decomposition"): 
                render_decomposition_tab(shared_config)
                
            with gr.Tab("Evaluation"): 
                render_evaluation_tab(shared_config)

        with gr.Tab("Analysis"):
            render_penta_comprehensive_analysis_tab(shared_config)

    with gr.Row():
        gr.HTML(
            "<div style='text-align: center; margin-top: 60px; padding-top: 24px; "
            "border-top: 1px solid #e5e5e5; color: #888888; font-size: 0.85rem; width: 100%;'>"
            "Designed and developed for PENTA Evaluation Suite (c) 2026. "
            "</div>"
        )

print("[PENTA] UI components initialized successfully.")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False
    )