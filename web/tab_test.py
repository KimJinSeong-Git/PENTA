# web/tab_test.py
import os
import requests
import gradio as gr

# Function to fetch loaded values from the global shared state
def display_shared_values(config):
    # Masking logic for Hugging Face Token
    hf_token = config.get('hf_token', '')
    masked_hf = f"Loaded & Masked (...{hf_token[-4:]})" if hf_token else "Empty"
    
    # Format the shared dictionary into a readable Markdown string
    markdown_content = f"""
    ### Current Shared Configurations
    
    * **OpenRouter API Key**: `{ "Loaded & Masked (..." + config['api_key'][-4:] + ")" if config.get('api_key') else "Empty" }`
    * **Hugging Face Token**: `{masked_hf}`
    
    | Role | Selected Model | Temperature | Top-p | Max Tokens | Est. Calls |
    | :--- | :--- | :---: | :---: | :---: | :---: |
    | **Target LM** | `{config.get('target_model', 'N/A')}` | `{config.get('target_temp', 'N/A')}` | `{config.get('target_top_p', 'N/A')}` | `{config.get('target_tokens', 'N/A')}` | `{config.get('t_runs', 'N/A')}` |
    | **Attacker LM** | `{config.get('attacker_model', 'N/A')}` | `{config.get('attacker_temp', 'N/A')}` | `{config.get('attacker_top_p', 'N/A')}` | `{config.get('attacker_tokens', 'N/A')}` | `{config.get('a_runs', 'N/A')}` |
    | **Judgment LM** | `{config.get('judgment_model', 'N/A')}` | `{config.get('judgment_temp', 'N/A')}` | `{config.get('judgment_top_p', 'N/A')}` | `{config.get('judgment_tokens', 'N/A')}` | `{config.get('j_runs', 'N/A')}` |
    | **Decomposer LM** | `{config.get('decomposer_model', 'N/A')}` | `{config.get('decomposer_temp', 'N/A')}` | `{config.get('decomposer_top_p', 'N/A')}` | `{config.get('decomposer_tokens', 'N/A')}` | `{config.get('d_runs', 'N/A')}` |
    """
    return markdown_content

# Action to test OpenRouter connection using the shared API Key
def test_openrouter_connection(config):
    api_key = config.get("api_key", "").strip()
    if not api_key:
        return "[Connection Failed] API Key is empty. Please set it in the 'Settings' tab first."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Simple request to fetch available models to verify key validity
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            model_count = len(models_data)
            return f"[Connection Successful] Successfully retrieved {model_count} models from OpenRouter API."
        else:
            return f"[Connection Failed] Status {response.status_code}: {response.text}"
    except Exception as e:
        return f"[Connection Error] {str(e)}"

# Render function for the Test Tab
def render_test_tab(shared_config):
    gr.Markdown("## Shared State & Connection Test")
    gr.Markdown("Verify that configuration modifications in the 'Settings' tab are correctly synchronized across tabs.")

    # Box 1: Dynamic Config Monitor
    with gr.Group():
        state_monitor = gr.Markdown(value="*Click 'Refresh Monitor' to load active configurations.*")
        refresh_btn = gr.Button("Refresh Monitor", variant="primary")

    # Box 2: Live API Ping Test
    with gr.Group():
        gr.Markdown("### OpenRouter Ping Test")
        ping_btn = gr.Button("Test API Connection")
        ping_status = gr.Markdown(value="*Status: Idle*")

    # Bind interactions and pass the global shared_config State
    refresh_btn.click(
        fn=display_shared_values,
        inputs=shared_config,
        outputs=state_monitor
    )

    ping_btn.click(
        fn=test_openrouter_connection,
        inputs=shared_config,
        outputs=ping_status
    )