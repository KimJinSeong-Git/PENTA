import gradio as gr
from datasets import load_dataset

pipeline_active = {"status": True}
cached_dataset = {"data": None}

DATASET_PRESETS = {
    "JailbreakBench/JBB-Behaviors": {"prompt": "Goal", "label": "Category"},
    "walledai/HarmBench": {"prompt": "prompt", "label": "category"},
    "walledai/StrongREJECT": {"prompt": "prompt", "label": "category"},
    "sorry-bench/sorry-bench-202503": {"prompt": "turns", "label": "category"},
    "TrustAIRLab/in-the-wild-jailbreak-prompts": {"prompt": "prompt", "label": "source"},
    # "walledai/WildJailbreak_safe": {"prompt": "prompt", "label": "label"},
    # "walledai/WildJailbreak_unsafe": {"prompt": "prompt", "label": "label"},
    # "walledai/XSTest_safe": {"prompt": "prompt", "label": "label"},
    # "walledai/XSTest_unsafe": {"prompt": "prompt", "label": "label"},
    # "lmsys/toxic-chat_safe": {"prompt": "user_input", "label": "toxicity"},
    # "lmsys/toxic-chat_unsafe": {"prompt": "user_input", "label": "toxicity"},
    # "walledai/WildGuardTest_safe": {"prompt": "prompt", "label": "label"},
    # "walledai/WildGuardTest_unsafe": {"prompt": "prompt", "label": "label"},
    # "yahma/alpaca-cleaned": {"prompt": "instruction", "label": "label"} 
}

def update_dataset_columns(dataset_name):
    if dataset_name in DATASET_PRESETS:
        preset = DATASET_PRESETS[dataset_name]
        return preset["prompt"], preset["label"]
    return gr.update(), gr.update()

def load_and_extract_labels(shared_config, dataset_name, label_col):
    hf_token = shared_config.get("hf_token", "").strip()
    status_msg = f"[*] Loading dataset '{dataset_name}' to extract target categories...\n"
    
    try:
        kwargs = {"token": hf_token if hf_token else None}
        
        # Parse dataset name and derived label type
        base_name = dataset_name
        label_type = None
        if "_" in dataset_name:
            base_name, label_type = dataset_name.rsplit("_", 1)

        if "JBB-Behaviors" in dataset_name:
            ds = load_dataset(dataset_name, "behaviors", **kwargs)["harmful"]
            
        elif "HarmBench" in dataset_name:
            ds = load_dataset(dataset_name, "contextual", **kwargs)["train"]

        elif "in-the-wild-jailbreak-prompts" in dataset_name:
            ds = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_05_07', split='train')
            ds = ds["train"] if "train" in ds else ds
            ds = ds.map(lambda x: {label_col: "jailbreak"})
            
        elif "XSTest" in dataset_name:
            ds = load_dataset(base_name, **kwargs)
            ds = ds["test"] if "test" in ds else ds["train"]
            if label_type:
                ds = ds.filter(lambda x: x[label_col] == label_type)
                
        elif "WildJailbreak" in dataset_name:
            target_label = "adversarial_benign" if label_type == "safe" else "adversarial_harmful"
            ds = load_dataset(base_name, **kwargs)
            ds = ds["train"] if "train" in ds else ds
            ds = ds.filter(lambda x: x[label_col] == target_label)
            
        elif "toxic-chat" in dataset_name:
            target_label = 0 if label_type == "safe" else 1
            ds = load_dataset(base_name, "toxicchat0124", **kwargs)
            ds = ds["train"] if "train" in ds else ds
            ds = ds.filter(lambda x: x[label_col] == target_label)

        elif "WildGuardTest" in dataset_name:
            target_label = "unharmful" if label_type == "safe" else "harmful"
            ds = load_dataset(base_name, **kwargs)
            ds = ds["train"] if "train" in ds else ds
            ds = ds.filter(lambda x: x[label_col] == target_label)
            
        else:
            ds = load_dataset(dataset_name, **kwargs)
            ds = ds["train"] if "train" in ds else ds
            if "alpaca-cleaned" in dataset_name:
                ds = ds.select(range(min(1000, len(ds))))
                ds = ds.add_column(label_col, ["benign"] * len(ds))
        
        if label_col not in ds.column_names:
            raise ValueError(f"Column '{label_col}' does not exist in the dataset.")

        print(ds)
        cached_dataset["data"] = ds
        
        unique_labels = sorted({val for val in ds[label_col] if val is not None})
        
        status_msg += f"[+] Successfully loaded {len(ds)} rows.\n"
        status_msg += f"[+] Found {len(unique_labels)} unique labels. Select the targets to process below."
        
        return status_msg, gr.update(choices=unique_labels, value=unique_labels, visible=True)
        
    except Exception as e:
        status_msg += f"[Error] Failed to load dataset: {str(e)}\n"
        return status_msg, gr.update(choices=[], value=[], visible=False)