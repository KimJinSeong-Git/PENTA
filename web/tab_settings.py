# web/tab_settings.py
import os
import yaml
import gradio as gr
from core.config import DEFAULT_MODELS

YAML_FILE = "models.yaml"

def load_models_from_yaml():
    if not os.path.exists(YAML_FILE):
        initial_data = {"models": DEFAULT_MODELS.copy()}
        with open(YAML_FILE, "w", encoding="utf-8") as f:
            f.write("# PENTA Evaluation Suite - Supported Model Pool\n")
            yaml.safe_dump(initial_data, f, default_flow_style=False, allow_unicode=True)
        return DEFAULT_MODELS.copy()
    
    try:
        with open(YAML_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "models" in data and isinstance(data["models"], dict):
                return data["models"]
    except Exception as e:
        print(f"[Warning] Failed to read {YAML_FILE}: {e}")
        
    return DEFAULT_MODELS.copy()

def get_model_table_data(models_dict):
    return [
        [model_name, f"${meta.get('input_cost_1m', 0.0):.4f}", f"${meta.get('output_cost_1m', 0.0):.4f}"]
        for model_name, meta in models_dict.items()
    ]

def calculate_estimated_cost(t_mod, t_tok, t_runs, a_mod, a_tok, a_runs, j_mod, j_tok, j_runs, d_mod, d_tok, d_runs, models_dict):
    total_cost = 0.0
    
    def get_unit_cost(model_name, max_tokens):
        if model_name in models_dict:
            meta = models_dict[model_name]
            in_c = float(meta.get("input_cost_1m", 0.0))
            out_c = float(meta.get("output_cost_1m", 0.0))
            tokens_used = max_tokens * 2
            return ((tokens_used / 1_000_000) * (in_c + out_c))
        return 0.0

    total_cost += get_unit_cost(t_mod, t_tok) * float(t_runs)
    total_cost += get_unit_cost(a_mod, a_tok) * float(a_runs)
    total_cost += get_unit_cost(j_mod, j_tok) * float(j_runs)
    total_cost += get_unit_cost(d_mod, d_tok) * float(d_runs)
    
    return f"${total_cost:.4f} USD"

def save_api_key(key, current_config):
    cleaned_key = key.strip()
    if cleaned_key:
        current_config["api_key"] = cleaned_key
        return "[Success] OpenRouter API Key successfully loaded into session memory.", current_config
    return "[Warning] Please enter a valid non-empty API Key.", current_config

def save_hf_token(token, current_config):
    cleaned_token = token.strip()
    if cleaned_token:
        current_config["hf_token"] = cleaned_token
        return "[Success] Hugging Face Token successfully loaded into session memory.", current_config
    return "[Warning] Please enter a valid non-empty Hugging Face Token.", current_config

def sync_settings_to_shared_config(
    api_key, hf_token, t_mod, t_temp, t_p, t_tok, t_calls, a_mod, a_temp, a_p, a_tok, a_calls,
    j_mod, j_temp, j_p, j_tok, j_calls, d_mod, d_temp, d_p, d_tok, d_calls, current_config
):
    current_config.update({
        "api_key": api_key,
        "hf_token": hf_token,
        "target_model": t_mod, "target_temp": t_temp, "target_top_p": t_p, "target_tokens": t_tok, "t_runs": t_calls,
        "attacker_model": a_mod, "attacker_temp": a_temp, "attacker_top_p": a_p, "attacker_tokens": a_tok, "a_runs": a_calls,
        "judgment_model": j_mod, "judgment_temp": j_temp, "judgment_top_p": j_p, "judgment_tokens": j_tok, "j_runs": j_calls,
        "decomposer_model": d_mod, "decomposer_temp": d_temp, "decomposer_top_p": d_p, "decomposer_tokens": d_tok, "d_runs": d_calls
    })
    return current_config

def render_settings_tab(shared_config):
    models_dict = load_models_from_yaml()
    loaded_models_list = list(models_dict.keys())
    initial_value = "google/gemma-4-31b-it"
    
    initial_api_key = shared_config.value.get("api_key", "")
    initial_hf_token = shared_config.value.get("hf_token", "")
    
    current_models_state = gr.State(value=models_dict)

    gr.Markdown("## System Configuration")
    gr.Markdown("Configure individual models, credentials, and track your evaluation execution budget.")
    
    with gr.Group():
        gr.Markdown("### API Authentication & Access Tokens")
        
        with gr.Row():
            api_key_input = gr.Textbox(
                label="OpenRouter API Key",
                placeholder="sk-or-v1-...",
                value=initial_api_key,
                type="password",
                interactive=True,
                scale=4
            )
            api_key_btn = gr.Button("Apply Key", scale=1)
        api_key_status = gr.Markdown(
            value="*API Key is loaded.*" if initial_api_key else "*No active API Key found.*"
        )
        
        with gr.Row():
            hf_token_input = gr.Textbox(
                label="Hugging Face Token",
                placeholder="hf_...",
                value=initial_hf_token,
                type="password",
                interactive=True,
                scale=4
            )
            hf_token_btn = gr.Button("Apply Token", scale=1)
        hf_token_status = gr.Markdown(
            value="*Hugging Face Token is loaded.*" if initial_hf_token else "*No active HF Token found.*"
        )

    gr.Markdown("---")
    gr.Markdown("### Role-based Hyperparameters")

    with gr.Row():
        with gr.Column(variant="panel"):
            gr.Markdown("#### 1. Target LM")
            target_model = gr.Dropdown(label="Model", choices=loaded_models_list, value=initial_value, interactive=True)
            target_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True)
            target_top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=0.9, step=0.05, interactive=True)
            target_tokens = gr.Slider(label="Max Tokens", minimum=64, maximum=4096, value=2048, step=64, interactive=True)

        with gr.Column(variant="panel"):
            gr.Markdown("#### 2. Attacker LM")
            attacker_model = gr.Dropdown(label="Model", choices=loaded_models_list, value=initial_value, interactive=True)
            attacker_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True)
            attacker_top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=0.9, step=0.05, interactive=True)
            attacker_tokens = gr.Slider(label="Max Tokens", minimum=64, maximum=4096, value=512, step=64, interactive=True)

    with gr.Row():
        with gr.Column(variant="panel"):
            gr.Markdown("#### 3. Judgment LM")
            judgment_model = gr.Dropdown(label="Model", choices=loaded_models_list, value=initial_value, interactive=True)
            judgment_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True)
            judgment_top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=1.0, step=0.05, interactive=True)
            judgment_tokens = gr.Slider(label="Max Tokens", minimum=16, maximum=2048, value=2048, step=16, interactive=True)

        with gr.Column(variant="panel"):
            gr.Markdown("#### 4. Decomposer LM")
            decomposer_model = gr.Dropdown(label="Model", choices=loaded_models_list, value=initial_value, interactive=True)
            decomposer_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True)
            decomposer_top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=1.0, step=0.05, interactive=True)
            decomposer_tokens = gr.Slider(label="Max Tokens", minimum=64, maximum=4096, value=2048, step=64, interactive=True)

    with gr.Group():
        gr.Markdown("### Experiment Cost Predictor")
        with gr.Row():
            t_runs = gr.Number(label="Target LM Calls", value=1000, precision=0, interactive=True)
            a_runs = gr.Number(label="Attacker LM Calls", value=1000, precision=0, interactive=True)
            j_runs = gr.Number(label="Judgment LM Calls", value=1000, precision=0, interactive=True)
            d_runs = gr.Number(label="Decomposer LM Calls", value=1000, precision=0, interactive=True)
            
        with gr.Row():
            cost_display = gr.HTML()

    gr.Markdown("---")
    with gr.Group():
        gr.Markdown("### Supported Models & Price Specs")
        model_table = gr.Dataframe(
            headers=["Model Name", "Input Cost / 1M", "Output Cost / 1M"],
            datatype=["str", "str", "str"],
            value=get_model_table_data(models_dict),
            interactive=False
        )

    def update_cost_ui(t_mod, t_tok, t_calls, a_mod, a_tok, a_calls, j_mod, j_tok, j_calls, d_mod, d_tok, d_calls, current_dict):
        cost_str = calculate_estimated_cost(
            t_mod, t_tok, t_calls, a_mod, a_tok, a_calls, j_mod, j_tok, j_calls, d_mod, d_tok, d_calls, current_dict
        )
        return (
            f"<div style='text-align: right; width: 100%; font-size: 1.1rem; color: #333333; margin-top: 12px;'>"
            f"Estimated Experiment Cost: <span style='font-weight: 800; font-size: 1.6rem; color: #111111; "
            f"margin-left: 8px;'>{cost_str}</span></div>"
        )

    api_key_btn.click(
        fn=save_api_key,
        inputs=[api_key_input, shared_config],
        outputs=[api_key_status, shared_config]
    )

    hf_token_btn.click(
        fn=save_hf_token,
        inputs=[hf_token_input, shared_config],
        outputs=[hf_token_status, shared_config]
    )

    sync_inputs = [
        api_key_input, hf_token_input,
        target_model, target_temp, target_top_p, target_tokens, t_runs,
        attacker_model, attacker_temp, attacker_top_p, attacker_tokens, a_runs,
        judgment_model, judgment_temp, judgment_top_p, judgment_tokens, j_runs,
        decomposer_model, decomposer_temp, decomposer_top_p, decomposer_tokens, d_runs
    ]

    cost_inputs = [
        target_model, target_tokens, t_runs,
        attacker_model, attacker_tokens, a_runs,
        judgment_model, judgment_tokens, j_runs,
        decomposer_model, decomposer_tokens, d_runs,
        current_models_state
    ]
    
    for comp in cost_inputs[:-1]:
        comp.change(
            fn=update_cost_ui,
            inputs=cost_inputs,
            outputs=cost_display
        )

    for comp in sync_inputs:
        comp.change(
            fn=sync_settings_to_shared_config,
            inputs=sync_inputs + [shared_config],
            outputs=shared_config
        )
    
    cost_display.value = update_cost_ui(
        initial_value, 2048, 1000, initial_value, 512, 1000, initial_value, 64, 1000, initial_value, 2048, 1000, models_dict
    )