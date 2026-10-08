# web/tab_decomposition.py
import os
import json
import time
import gradio as gr

from core.prompt_utils import (
    extract_json, extract_decomposition_components, parse_categories,
    update_categories, save_decomposition_results, save_categories, get_response
)
from core.system_prompt import get_decomposer_prompt
from core.dataset_utils import DATASET_PRESETS, cached_dataset, update_dataset_columns, load_and_extract_labels
from core.config import DECOMP_DIR

def stop_decomposition_pipeline(shared_config):
    """Disable pipeline flag when cancellation is requested."""
    shared_config["decomp_active"] = False
    return "[Cancelled] Pipeline cancellation requested. Stopping at the next iteration...", shared_config

def run_decomposition_pipeline(
    shared_config, dataset_name, prompt_col, label_col, base_cat_path, selected_labels
):
    shared_config["decomp_active"] = True

    api_key = shared_config.get("api_key", "").strip()
    model_id = shared_config.get("decomposer_model", "").strip()
    max_tokens = shared_config.get("decomposer_tokens", 2048)
    temp = shared_config.get("decomposer_temp", 0.0)
    top_p = shared_config.get("decomposer_top_p", 1.0)

    if not api_key:
        yield "[Error] Missing API Key. Check Settings tab.", gr.update(), gr.update(), shared_config
        return
    if not model_id:
        yield "[Error] Missing Model. Check Settings tab.", gr.update(), gr.update(), shared_config
        return
    if not selected_labels:
        yield "[Warning] No labels selected.", gr.update(), gr.update(), shared_config
        return

    model_short = model_id.split('/')[-1]
    dataset_short = dataset_name.split('/')[-1]
    
    category_file_path = f"{DECOMP_DIR}/[category] {model_short}_{dataset_short}.json"
    results_file_path = f"{DECOMP_DIR}/[results] {model_short}_{dataset_short}.json"

    console_log = "[*] Launching Decomposer Pipeline...\n"
    records_stream_text = ""
    taxonomy_summary_text = "Taxonomy Status: Uninitialized"
    
    yield console_log, records_stream_text, taxonomy_summary_text, shared_config

    total_categories = {"roles": [], "domains": [], "actions": [], "objects": [], "formats": []}
    
    if os.path.exists(category_file_path):
        try:
            with open(category_file_path, "r", encoding="utf-8") as f:
                total_categories = json.load(f)
            console_log += f"[*] Loaded existing ontology from '{category_file_path}'.\n"
        except Exception as e:
            console_log += f"[Error] Error reading category file: {e}\n"
    elif os.path.exists(base_cat_path):
        try:
            with open(base_cat_path, "r", encoding="utf-8") as f:
                total_categories = json.load(f)
            console_log += f"[*] Loaded base categories from '{base_cat_path}'.\n"
        except Exception as e:
            console_log += f"[Error] Error reading '{base_cat_path}': {e}\n"
    else:
        console_log += f"[*] Ontology file not found. Initializing empty.\n"
    
    ds = cached_dataset["data"]
    if ds is None:
        console_log += "[Error] Dataset cache is empty. Please load dataset first.\n"
        yield console_log, records_stream_text, taxonomy_summary_text, shared_config
        return

    selected_labels = [str(lbl) for lbl in selected_labels]
    filtered_ds = []
    for row in ds:
        raw_lbl = str(row.get(label_col))
        lbl_str = str(raw_lbl[0]) if isinstance(raw_lbl, list) and len(raw_lbl) > 0 else str(raw_lbl) if raw_lbl else ""
        if lbl_str in selected_labels:
            filtered_ds.append(row)

    decomp_results = {
        "metadata": {
            "dataset": dataset_name, "decomposer": model_id,
            "savetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
        "records": []
    }
    
    processed_prompts = set()

    if os.path.exists(results_file_path):
        try:
            with open(results_file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            
            existing_records = existing_data.get("records", [])
            decomp_results["records"] = existing_records

            for r in existing_records:
                p_text = r.get("prompt", "")
                p_text = str(p_text[0]) if isinstance(p_text, list) and len(p_text) > 0 else str(p_text)
                if p_text: processed_prompts.add(p_text)

                r_idx = r.get("idx", "N/A")
                r_label = r.get("label", "Unknown")
                r_cats = r.get("categories", {})

                old_log = f"=== [RECORD #{r_idx}] [{r_label}] (LOADED FROM CACHE) ===\n"
                old_log += f"Label: {r_label}\nPrompt: {p_text}\nCategories Extracted:\n"
                old_log += f"  - Roles: {r_cats.get('roles', [])}\n  - Domains: {r_cats.get('domains', [])}\n"
                old_log += f"  - Actions: {r_cats.get('actions', [])}\n  - Objects: {r_cats.get('objects', [])}\n"
                old_log += f"  - Formats: {r_cats.get('formats', [])}\n\n"
                records_stream_text = old_log + records_stream_text

            console_log += f"[Resume Mode] Found existing results file!\n"
            console_log += f"   Loaded {len(existing_records)} completed records.\n"
        except Exception as e:
            console_log += f"[Warning] Failed to parse existing result file ({e}). Starting fresh...\n"

    remaining_count = sum(1 for r in filtered_ds if (str(r.get(prompt_col, [""])[0]) if isinstance(r.get(prompt_col), list) else str(r.get(prompt_col, ""))) not in processed_prompts)

    console_log += f"[+] Active targets: {len(filtered_ds)} rows (Remaining: {remaining_count}).\n"
    
    taxonomy_summary_text = "==================================================\nGLOBAL TAXONOMY SHARE STATUS\n==================================================\n"
    for key in ["roles", "domains", "actions", "objects", "formats"]:
        items = total_categories.get(key, [])
        taxonomy_summary_text += f"{key.upper()} (Total: {len(items)})\n   - {sorted(items)}\n\n"

    yield console_log, records_stream_text, taxonomy_summary_text, shared_config

    for i, row in enumerate(filtered_ds):
        if not shared_config.get("decomp_active", True):
            console_log += "\n[Stopped] Pipeline execution stopped by the user.\n"
            yield console_log, records_stream_text, taxonomy_summary_text, shared_config
            return

        raw_prompt = row.get(prompt_col)
        prompt = str(raw_prompt[0]) if isinstance(raw_prompt, list) and len(raw_prompt) > 0 else str(raw_prompt)

        raw_label = row.get(label_col)
        label = str(raw_label[0]) if isinstance(raw_label, list) and len(raw_label) > 0 else str(raw_label)

        if prompt in processed_prompts:
            console_log += f"[{i+1}/{len(filtered_ds)}] Skipping completed prompt: '{label}'\n"
            yield console_log, records_stream_text, taxonomy_summary_text, shared_config
            continue

        console_log += f"\n[{i+1}/{len(filtered_ds)}] Processing: '{label}'\n"
        yield console_log, records_stream_text, taxonomy_summary_text, shared_config

        decomp_msg = [
            {"role": "system", "content": get_decomposer_prompt(
                total_categories.get("roles", []), total_categories.get("domains", []),
                total_categories.get("actions", []), total_categories.get("objects", []),
                total_categories.get("formats", []), is_addition=True
            )},
            {"role": "user", "content": prompt}
        ]

        try:
            decomp_response = get_response(api_key, model_id, decomp_msg, max_tokens, temp, top_p)
        except Exception as e:
            console_log += f"[Warning] API Request Error: {e}. Skipping...\n"
            yield console_log, records_stream_text, taxonomy_summary_text, shared_config
            continue

        if not shared_config.get("decomp_active", True):
            yield console_log + "\n[Stopped] after API call.\n", records_stream_text, taxonomy_summary_text, shared_config
            return

        try:
            decomp_json = extract_json(decomp_response)
        except Exception:
            decomp_json = None

        if decomp_json is None:
            console_log += "[Warning] Failed to parse valid JSON. Skipping...\n"
            yield console_log, records_stream_text, taxonomy_summary_text, shared_config
            continue

        components = extract_decomposition_components(decomp_json)
        categories = parse_categories(components)

        record = {"idx": i, "label": label, "prompt": prompt, "categories": categories}
        decomp_results["records"].append(record)
        decomp_results["records"].sort(key=lambda x: int(x.get("idx", 0)))
        processed_prompts.add(prompt)

        decomp_results["metadata"]["savetime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        save_decomposition_results(decomp_results, results_file_path)
        console_log += f"[Saved] Appended & Sorted to '{results_file_path}'\n"

        is_change, total_categories = update_categories(total_categories, components)
        if is_change:
            save_categories(is_change, total_categories, category_file_path)
            console_log += f"[Updated] Ontology updated -> '{category_file_path}'\n"

        new_record_log = f"=== [RECORD #{i}] [{label}] ===\nLabel: {label}\nPrompt: {prompt}\nCategories Extracted:\n"
        new_record_log += f"  - Roles: {categories.get('roles', [])}\n  - Domains: {categories.get('domains', [])}\n"
        new_record_log += f"  - Actions: {categories.get('actions', [])}\n  - Objects: {categories.get('objects', [])}\n"
        new_record_log += f"  - Formats: {categories.get('formats', [])}\n\n"
        
        records_stream_text = new_record_log + records_stream_text

        taxonomy_summary_text = "==================================================\nGLOBAL TAXONOMY SHARE STATUS\n==================================================\n"
        for key in ["roles", "domains", "actions", "objects", "formats"]:
            items = total_categories.get(key, [])
            taxonomy_summary_text += f"{key.upper()} (Total: {len(items)})\n   - {sorted(items)}\n\n"

        yield console_log, records_stream_text, taxonomy_summary_text, shared_config

    save_decomposition_results(decomp_results, results_file_path)
    console_log += "\n[Complete] Pipeline process completed!"
    yield console_log, records_stream_text, taxonomy_summary_text, shared_config

def render_decomposition_tab(shared_config):
    gr.Markdown("## Semantic Decomposition Pipeline")
    gr.Markdown("Decompose safety benchmark prompts into isolated Roles, Goals, and Formats according to the PENTA schema.")

    with gr.Accordion("Dataset & Directory Specifications", open=True):
        with gr.Row():
            dataset_name_input = gr.Dropdown(
                label="HF Dataset Path", choices=list(DATASET_PRESETS.keys()),
                value="JailbreakBench/JBB-Behaviors", allow_custom_value=True, interactive=True, scale=3
            )
            prompt_column_input = gr.Textbox(label="Prompt Column Name", value="turns", interactive=True, scale=1)
            label_column_input = gr.Textbox(label="Label Column Name", value="category", interactive=True, scale=1)
            base_category_path_input = gr.Textbox(label="Base Ontology File Path (.json)", value="./base.json", interactive=True, scale=2)
            load_dataset_btn = gr.Button("Load Metadata", scale=1, variant="secondary")

    label_checkboxes = gr.CheckboxGroup(label="Filter Targets", choices=[], value=[], interactive=True, visible=False)

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Pipeline Control")
        with gr.Column(scale=2):
            with gr.Row():
                start_btn = gr.Button("Start Pipeline", variant="primary", scale=2)
                stop_btn = gr.Button("Stop", variant="stop", scale=1)

    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            with gr.Tabs():
                with gr.Tab("Process Console Stream"):
                    console_output = gr.Textbox(label="Process Console Stream", lines=18, max_lines=22, interactive=False)

        with gr.Column(scale=2, variant="panel"):
            with gr.Tabs():
                with gr.Tab("Extracted Records Stream"):
                    records_viewer = gr.Textbox(label="Live Extracted Prompts Log", lines=18, max_lines=22, interactive=False)
                with gr.Tab("Global Taxonomy Summary"):
                    ontology_viewer = gr.Textbox(label="Unified Taxonomy Framework Aggregation", lines=18, max_lines=22, interactive=False)

    dataset_name_input.change(fn=update_dataset_columns, inputs=[dataset_name_input], outputs=[prompt_column_input, label_column_input])
    load_dataset_btn.click(fn=load_and_extract_labels, inputs=[shared_config, dataset_name_input, label_column_input], outputs=[console_output, label_checkboxes])
    
    start_btn.click(
        fn=run_decomposition_pipeline,
        inputs=[shared_config, dataset_name_input, prompt_column_input, label_column_input, base_category_path_input, label_checkboxes],
        outputs=[console_output, records_viewer, ontology_viewer, shared_config]
    )
    
    stop_btn.click(
        fn=stop_decomposition_pipeline, 
        inputs=[shared_config], 
        outputs=[console_output, shared_config]
    )