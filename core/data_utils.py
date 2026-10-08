# core/data_utils.py
import os
import glob
import json
import pandas as pd
from core.config import EVAL_DIR

def get_eval_files():
    if not os.path.exists(EVAL_DIR):
        return []
    return [os.path.basename(f) for f in glob.glob(os.path.join(EVAL_DIR, "*.json"))]

def parse_eval_files():
    files = get_eval_files()
    parsed_data = []
    
    for filename in files:
        filepath = os.path.join(EVAL_DIR, filename)
        try:
            name_body = filename
            if name_body.startswith("[eval]"):
                name_body = name_body.replace("[eval]", "", 1).strip()
            if name_body.endswith(".json"):
                name_body = name_body[:-5].strip()
                
            if "_" in name_body:
                model, dataset = name_body.rsplit("_", 1)
            else:
                model = name_body
                dataset = "Unknown"
                
            parsed_data.append({
                "filename": filename,
                "model": model.strip(),
                "dataset": dataset.strip(),
                "filepath": filepath
            })
        except:
            continue
    return parsed_data

def get_model_choices():
    data = parse_eval_files()
    return sorted(list(set(item["model"] for item in data)))

def get_dataset_choices(selected_model):
    if not selected_model:
        return []
    data = parse_eval_files()
    return sorted(list(set(item["dataset"] for item in data if item["model"] == selected_model)))

def get_all_dataset_choices():
    data = parse_eval_files()
    return sorted(list(set(item["dataset"] for item in data)))

def get_models_for_dataset(dataset):
    data = parse_eval_files()
    return sorted(list(set(item["model"] for item in data if item["dataset"] == dataset)))

def truncate_text(text, max_len=100):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text[:max_len] + "..." if len(text) > max_len else text

def load_and_process_eval_data(model, dataset):
    if not model or not dataset:
        return pd.DataFrame()
        
    data = parse_eval_files()
    target_item = next((item for item in data if item["model"] == model and item["dataset"] == dataset), None)
            
    if not target_item:
        return pd.DataFrame()
    
    try:
        with open(target_item["filepath"], "r", encoding="utf-8") as f:
            raw_json = json.load(f)
            
        if isinstance(raw_json, dict) and "records" in raw_json:
            records = raw_json["records"]
            df = pd.json_normalize(records)
            
            for idx, rec in enumerate(records):
                if "categories" in rec and isinstance(rec["categories"], dict):
                    df.at[idx, 'raw_categories'] = json.dumps(rec["categories"])
        else:
            return pd.DataFrame()

        rename_map = {
            'idx': 'Index',
            'label': 'Category',
            'prompt': 'Prompt',
            'target_response': 'Target Response',
            'evaluation.score': 'ASR',
            'evaluation.refusal': 'Refusal',
            'evaluation.specificity': 'Specificity',
            'evaluation.convincingness': 'Convincingness'
        }
        df = df.rename(columns=rename_map)

        num_cols = ['ASR', 'Refusal', 'Specificity', 'Convincingness']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        drop_cols = [c for c in df.columns if 'judge' in c.lower() or ('categories.' in c and c != 'raw_categories')]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        df['Model'] = model
        return df

    except Exception as e:
        print(f"Data Load Error for {model}: {e}")
        return pd.DataFrame()