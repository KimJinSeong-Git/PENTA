# core/penta_analytics.py
# PENTA Analytics Engine

from __future__ import annotations

import glob
import json
import os
import re
import itertools
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import fisher_exact
from scipy.spatial.distance import cosine, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.stats.multitest import multipletests

from core.config import EVAL_DIR, DECOMP_DIR

# Constants and taxonomy
MODULES = ["Role", "Domain", "Action", "Object", "Format"]
MODULE_KEYS = ["roles", "domains", "actions", "objects", "formats"]

MODEL_ORDER = [
    "claude-sonnet-4.5",
    "gpt-5",
    "gemini-3.1-pro-preview",
    "llama-3.1-8b-instruct",
    "qwen-2.5-7b-instruct",
    "gemma-3-12b-it",
]

MODEL_DISPLAY = {
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "gpt-5": "GPT-5",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "qwen-2.5-7b-instruct": "Qwen-2.5-7B",
    "gemma-3-12b-it": "Gemma-3-12B",
}

MODEL_TYPE = {
    "claude-sonnet-4.5": "Closed",
    "gpt-5": "Closed",
    "gemini-3.1-pro-preview": "Closed",
    "llama-3.1-8b-instruct": "Open",
    "qwen-2.5-7b-instruct": "Open",
    "gemma-3-12b-it": "Open",
}

DATASET_ORDER = ["HarmBench", "JBB-Behaviors", "in-the-wild-jailbreak-prompts"]
DATASET_DISPLAY = {
    "HarmBench": "HarmBench",
    "JBB-Behaviors": "JBB-Behaviors",
    "in-the-wild-jailbreak-prompts": "In-the-wild",
}

TYPE_COLORS = {"Closed": "#D32F2F", "Open": "#1976D2"}
MODEL_COLORS = {
    "Claude Sonnet 4.5": "#D32F2F",
    "GPT-5": "#E64A19",
    "Gemini 3.1 Pro": "#F57C00",
    "Llama-3.1-8B": "#1976D2",
    "Qwen-2.5-7B": "#0288D1",
    "Gemma-3-12B": "#0097A7",
}

FINGERPRINT_VIEWS = [
    "Single: Integrated (Full Structure)",
    "Single: Drop (Vulnerability)",
    "Single: Rise (Over-refusal)",
    "Pairwise: Integrated (Full Synergy)",
    "Pairwise: Drop (Synergy Vulnerability)",
    "Pairwise: Rise (Synergy Over-refusal)",
]

# Global cache
_CACHE: Dict[str, Any] = {}


def _get_eval_df() -> pd.DataFrame:
    """Load all records from evaluation_results into a unified DataFrame."""
    if "eval_df" in _CACHE:
        return _CACHE["eval_df"]

    files = glob.glob(os.path.join(EVAL_DIR, "*.json"))
    rows = []
    for filepath in files:
        fname = os.path.basename(filepath)
        if not fname.startswith("[eval]"):
            continue
        body = fname.replace("[eval]", "").replace(".json", "").strip()
        if "_" not in body:
            continue
        model_key, dataset_key = body.rsplit("_", 1)
        if model_key not in MODEL_ORDER or dataset_key not in DATASET_ORDER:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            records = data.get("records", [])
            for rec in records:
                eval_data = rec.get("evaluation", {})
                refusal = float(eval_data.get("refusal", 1.0))
                score = float(eval_data.get("score", 0.0))
                specificity = float(eval_data.get("specificity", 0.0))
                convincingness = float(eval_data.get("convincingness", 0.0))

                cats = rec.get("categories", {})
                tags = []
                for mod_key in MODULE_KEYS:
                    mod_name = mod_key[:-1].capitalize()
                    for item in cats.get(mod_key, []):
                        if item and str(item).strip():
                            tags.append(f"{mod_name}:{str(item).strip()}")

                raw_label = rec.get("label", "Unknown")
                if isinstance(raw_label, list):
                    raw_label = raw_label[0] if raw_label else "Unknown"
                raw_label = str(raw_label).strip()

                prompt_text = rec.get("prompt", "")
                if isinstance(prompt_text, list):
                    prompt_text = prompt_text[0] if prompt_text else ""

                target_resp = rec.get("target_response", "")

                rows.append({
                    "Model": model_key,
                    "Model_Display": MODEL_DISPLAY.get(model_key, model_key),
                    "Model_Type": MODEL_TYPE.get(model_key, "Unknown"),
                    "Dataset": dataset_key,
                    "Dataset_Display": DATASET_DISPLAY.get(dataset_key, dataset_key),
                    "Label": raw_label,
                    "Prompt": str(prompt_text),
                    "Target_Response": str(target_resp),
                    "Refusal": refusal,
                    "Score": score,
                    "Specificity": specificity,
                    "Convincingness": convincingness,
                    "Tags": list(set(tags)),
                    "Num_Components": len(set(tags)),
                })
        except Exception as e:
            print(f"[Warning] Failed to parse {filepath}: {e}")

    df = pd.DataFrame(rows)
    _CACHE["eval_df"] = df
    return df


def _get_decomp_data() -> Dict[str, dict]:
    """Load decomposition results files."""
    if "decomp_data" in _CACHE:
        return _CACHE["decomp_data"]

    data = {}
    for ds_key in DATASET_ORDER:
        path = os.path.join(DECOMP_DIR, f"[results] gemma-4-31b-it_{ds_key}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data[ds_key] = json.load(fp)
            except Exception as e:
                print(f"[Warning] Failed to read {path}: {e}")
    _CACHE["decomp_data"] = data
    return data


# Corpus structure and module distribution

def get_table1_corpus_structure() -> pd.DataFrame:
    """Compute Table 1: Corpus structure by source."""
    decomp = _get_decomp_data()
    rows = []

    for ds_key in DATASET_ORDER:
        ds_name = DATASET_DISPLAY.get(ds_key, ds_key)
        records = decomp.get(ds_key, {}).get("records", [])
        if not records:
            continue

        r_counts, d_counts, a_counts, o_counts, f_counts = [], [], [], [], []
        r_unique, d_unique, a_unique, o_unique, f_unique = set(), set(), set(), set(), set()
        active_3, active_4, active_5 = 0, 0, 0

        for r in records:
            cats = r.get("categories", {})
            roles = [x for x in cats.get("roles", []) if x]
            domains = [x for x in cats.get("domains", []) if x]
            actions = [x for x in cats.get("actions", []) if x]
            objects = [x for x in cats.get("objects", []) if x]
            formats = [x for x in cats.get("formats", []) if x]

            r_counts.append(len(roles))
            d_counts.append(len(domains))
            a_counts.append(len(actions))
            o_counts.append(len(objects))
            f_counts.append(len(formats))

            r_unique.update(roles)
            d_unique.update(domains)
            a_unique.update(actions)
            o_unique.update(objects)
            f_unique.update(formats)

            act_cnt = sum([len(roles) > 0, len(domains) > 0, len(actions) > 0, len(objects) > 0, len(formats) > 0])
            if act_cnt == 3:
                active_3 += 1
            elif act_cnt == 4:
                active_4 += 1
            elif act_cnt >= 5:
                active_5 += 1

        n = len(records)
        rows.append({
            "Dataset": ds_name,
            "N": n,
            "Unique R": len(r_unique),
            "Unique D": len(d_unique),
            "Unique A": len(a_unique),
            "Unique O": len(o_unique),
            "Unique F": len(f_unique),
            "Mean R": round(float(np.mean(r_counts)), 2),
            "Mean D": round(float(np.mean(d_counts)), 2),
            "Mean A": round(float(np.mean(a_counts)), 2),
            "Mean O": round(float(np.mean(o_counts)), 2),
            "Mean F": round(float(np.mean(f_counts)), 2),
            "Total Mean": round(float(np.mean(r_counts) + np.mean(d_counts) + np.mean(a_counts) + np.mean(o_counts) + np.mean(f_counts)), 2),
            "Active 3 (%)": round(active_3 / n * 100, 1),
            "Active 4 (%)": round(active_4 / n * 100, 1),
            "Active 5 (%)": round(active_5 / n * 100, 1),
        })

    return pd.DataFrame(rows)


def get_component_catalog(dataset_key: str) -> Dict[str, List[str]]:
    """Return dictionary of components by module for the selected dataset."""
    decomp = _get_decomp_data()
    records = decomp.get(dataset_key, {}).get("records", [])

    catalog = {m: set() for m in MODULES}
    for r in records:
        cats = r.get("categories", {})
        for mod_key in MODULE_KEYS:
            mod_name = mod_key[:-1].capitalize()
            for item in cats.get(mod_key, []):
                if item and str(item).strip():
                    catalog[mod_name].add(str(item).strip())

    return {m: sorted(list(catalog[m])) for m in MODULES}


def get_decomposed_samples(dataset_key: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return sample decomposed prompts matching Tables 12, 13, 14."""
    decomp = _get_decomp_data()
    records = decomp.get(dataset_key, {}).get("records", [])
    samples = []
    for r in records[:limit]:
        p = r.get("prompt", "")
        if isinstance(p, list):
            p = p[0] if p else ""
        cats = r.get("categories", {})
        samples.append({
            "Index": r.get("idx", 0),
            "Label": r.get("label", "N/A"),
            "Prompt": p,
            "Role": ", ".join(cats.get("roles", [])) or "(None)",
            "Domain": ", ".join(cats.get("domains", [])) or "(None)",
            "Action": ", ".join(cats.get("actions", [])) or "(None)",
            "Object": ", ".join(cats.get("objects", [])) or "(None)",
            "Format": ", ".join(cats.get("formats", [])) or "(None)",
        })
    return samples


# Baseline safety evaluation
def get_table5_baseline_refusal(as_asr: bool = False) -> pd.DataFrame:
    """Compute Table 5: Refusal rate (%) or Attack Success Rate (%) by model and source."""
    df = _get_eval_df()
    rows = []

    for model_key in MODEL_ORDER:
        mdf = df[df["Model"] == model_key]
        if mdf.empty:
            continue

        hb_ref = mdf[mdf["Dataset"] == "HarmBench"]["Refusal"].mean()
        jbb_ref = mdf[mdf["Dataset"] == "JBB-Behaviors"]["Refusal"].mean()
        wild_ref = mdf[mdf["Dataset"] == "in-the-wild-jailbreak-prompts"]["Refusal"].mean()
        all_ref = mdf["Refusal"].mean()

        if as_asr:
            # ASR = 100 - Refusal
            hb_val = (1.0 - hb_ref) * 100
            jbb_val = (1.0 - jbb_ref) * 100
            wild_val = (1.0 - wild_ref) * 100
            all_val = (1.0 - all_ref) * 100
        else:
            hb_val = hb_ref * 100
            jbb_val = jbb_ref * 100
            wild_val = wild_ref * 100
            all_val = all_ref * 100

        rows.append({
            "Type": MODEL_TYPE.get(model_key, "Unknown"),
            "Model": MODEL_DISPLAY.get(model_key, model_key),
            "HarmBench": round(hb_val, 1),
            "JBB-Behaviors": round(jbb_val, 1),
            "In-the-wild": round(wild_val, 1),
            "Overall": round(all_val, 1),
        })

    return pd.DataFrame(rows)


def get_table6_category_refusal(dataset_key: str = "HarmBench") -> pd.DataFrame:
    """Compute Table 6: Granular Refusal rate (%) by harm category."""
    df = _get_eval_df()
    sub = df[df["Dataset"] == dataset_key]
    if sub.empty:
        return pd.DataFrame()

    categories = sorted(sub["Label"].unique())
    rows = []

    for cat in categories:
        cat_df = sub[sub["Label"] == cat]
        row = {"Category": cat, "Count": len(cat_df) // len(MODEL_ORDER)}
        for m in MODEL_ORDER:
            m_sub = cat_df[cat_df["Model"] == m]
            if not m_sub.empty:
                row[MODEL_DISPLAY.get(m, m)] = round(m_sub["Refusal"].mean() * 100, 1)
            else:
                row[MODEL_DISPLAY.get(m, m)] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def plot_baseline_comparison(as_asr: bool = False) -> plt.Figure:
    """Plot grouped bar chart of Table 5."""
    t5 = get_table5_baseline_refusal(as_asr=as_asr)
    metric_label = "Attack Success Rate (ASR %)" if as_asr else "Refusal Rate (%)"

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
    datasets = ["HarmBench", "JBB-Behaviors", "In-the-wild", "Overall"]
    n_models = len(t5)
    width = 0.8 / n_models
    x = np.arange(len(datasets))

    for i, (_, row) in enumerate(t5.iterrows()):
        m_name = row["Model"]
        color = MODEL_COLORS.get(m_name, "#555555")
        vals = [row[ds] for ds in datasets]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=m_name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title(f"Baseline Defense Performance across Corpora ({metric_label})", fontsize=12, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout()
    return fig


# Structural vulnerability and synergy

def compute_single_module_ranking_per_model(min_support: int = 10) -> pd.DataFrame:
    """Compute refusal lift for each single component per model."""
    df = _get_eval_df()
    results = []

    for model_key, sub_df in df.groupby("Model"):
        baseline_ref = sub_df["Refusal"].mean()

        tag_records = []
        for _, row in sub_df.iterrows():
            for tag in row["Tags"]:
                cat, comp = tag.split(":", 1) if ":" in tag else ("Unknown", tag)
                tag_records.append({"Category": cat, "Component": comp, "Refusal": row["Refusal"]})

        if not tag_records:
            continue

        tdf = pd.DataFrame(tag_records)
        stats = tdf.groupby(["Category", "Component"]).agg(
            Count=("Refusal", "count"),
            Mean_Refusal=("Refusal", "mean")
        ).reset_index()

        stats["Baseline_Refusal"] = baseline_ref
        # Lift Delta in percentage points: Mean - Baseline
        # Negative indicates refusal drop (high risk), Positive indicates refusal rise (over-refusal)
        stats["Delta_pp"] = (stats["Mean_Refusal"] - baseline_ref) * 100
        stats["Model"] = model_key
        stats["Model_Display"] = MODEL_DISPLAY.get(model_key, model_key)
        stats["Sufficient_Support"] = stats["Count"] >= min_support

        results.append(stats)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def get_table3_single_risk_ranking(
    model_name: str = "All",
    module_filter: str = "All",
    min_support: int = 10,
    top_k: int = 5
) -> pd.DataFrame:
    """Get high-risk (drop) and low-risk (rise) components matching Table 3."""
    all_ranks = compute_single_module_ranking_per_model(min_support=min_support)
    if all_ranks.empty:
        return pd.DataFrame()

    sub = all_ranks[all_ranks["Count"] >= min_support].copy()

    if model_name != "All":
        sub = sub[sub["Model_Display"] == model_name]

    if module_filter != "All":
        sub = sub[sub["Category"] == module_filter]

    rows = []
    for m, mdf in sub.groupby("Model_Display"):
        # High-risk: largest drop (most negative Delta_pp)
        drops = mdf.sort_values("Delta_pp", ascending=True).head(top_k)
        for rank, (_, r) in enumerate(drops.iterrows(), 1):
            rows.append({
                "Model": m,
                "Type": "High-risk (Drop)",
                "Rank": rank,
                "Module": r["Category"],
                "Component": r["Component"],
                "Delta (pp)": round(r["Delta_pp"], 2),
                "Mean Refusal (%)": round(r["Mean_Refusal"] * 100, 1),
                "Baseline (%)": round(r["Baseline_Refusal"] * 100, 1),
                "Support (n)": int(r["Count"]),
            })

        # Low-risk: largest rise (most positive Delta_pp)
        rises = mdf.sort_values("Delta_pp", ascending=False).head(top_k)
        for rank, (_, r) in enumerate(rises.iterrows(), 1):
            rows.append({
                "Model": m,
                "Type": "Low-risk (Rise)",
                "Rank": rank,
                "Module": r["Category"],
                "Component": r["Component"],
                "Delta (pp)": round(r["Delta_pp"], 2),
                "Mean Refusal (%)": round(r["Mean_Refusal"] * 100, 1),
                "Baseline (%)": round(r["Baseline_Refusal"] * 100, 1),
                "Support (n)": int(r["Count"]),
            })

    return pd.DataFrame(rows)


def plot_single_risk_chart(model_name: str, module_filter: str = "All", min_support: int = 10, top_k: int = 5) -> plt.Figure:
    """Plot horizontal bar chart of top Drop and Rise components for a model."""
    df_rank = get_table3_single_risk_ranking(
        model_name=model_name, module_filter=module_filter, min_support=min_support, top_k=top_k
    )

    if df_rank.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No components meet the criteria.", ha="center", va="center")
        ax.axis("off")
        return fig

    drops = df_rank[df_rank["Type"] == "High-risk (Drop)"].sort_values("Delta (pp)", ascending=True)
    rises = df_rank[df_rank["Type"] == "Low-risk (Rise)"].sort_values("Delta (pp)", ascending=False)

    plot_df = pd.concat([drops, rises])
    plot_df["Label"] = plot_df["Module"] + ": " + plot_df["Component"]

    fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.38)), dpi=120)
    colors = ["#D32F2F" if t == "High-risk (Drop)" else "#2CA02C" for t in plot_df["Type"]]

    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df["Delta (pp)"], color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Label"], fontsize=9.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Refusal Lift Delta (percentage points vs model baseline)", fontsize=10.5)
    ax.set_title(f"Component Vulnerability Surface: {model_name} (Top Drops & Rises)", fontsize=11.5, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        val = row["Delta (pp)"]
        offset = 1.0 if val >= 0 else -1.0
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}pp (n={row['Support (n)']})",
                va="center", ha=ha, fontsize=8)

    legend_elements = [
        Patch(facecolor="#D32F2F", label="Drop (Vulnerability Surface)"),
        Patch(facecolor="#2CA02C", label="Rise (Over-Refusal / Strict)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    return fig


def _compute_pairwise_synergy_cache(min_cooccur: int = 10, min_solo: int = 10) -> pd.DataFrame:
    """Precompute and cache pairwise synergy per model."""
    cache_key = f"syn_{min_cooccur}_{min_solo}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    cache_file = os.path.join(EVAL_DIR, f"penta_synergy_cache_{min_cooccur}_{min_solo}.json")
    if os.path.exists(cache_file):
        try:
            df_disk = pd.read_json(cache_file)
            _CACHE[cache_key] = df_disk
            return df_disk
        except Exception:
            pass

    df = _get_eval_df()
    rows = []

    for model_key, sub_df in df.groupby("Model"):
        sub_df = sub_df.reset_index(drop=True)
        all_tags = sorted({t for tags in sub_df["Tags"] for t in tags})
        tag_matrix = pd.DataFrame({tag: sub_df["Tags"].apply(lambda tags: tag in tags) for tag in all_tags})
        refusal = sub_df["Refusal"].values

        for tag_a, tag_b in itertools.combinations(all_tags, 2):
            has_a = tag_matrix[tag_a].values
            has_b = tag_matrix[tag_b].values
            both = has_a & has_b
            if both.sum() < min_cooccur:
                continue

            a_only = has_a & ~has_b
            b_only = has_b & ~has_a
            if a_only.sum() < min_solo or b_only.sum() < min_solo:
                continue

            ref_a = refusal[a_only].mean()
            ref_b = refusal[b_only].mean()
            obs = refusal[both].mean()
            exp = ref_a + ref_b - (ref_a * ref_b)
            score = exp - obs  # positive score = vulnerability synergy

            # 2x2 contingency table for Fisher's exact test
            n_both_ref = refusal[both].sum()
            n_both_fail = len(refusal[both]) - n_both_ref
            n_noboth_ref = refusal[~both].sum()
            n_noboth_fail = len(refusal[~both]) - n_noboth_ref

            table = [[n_both_ref, n_both_fail], [n_noboth_ref, n_noboth_fail]]
            _, pval = fisher_exact(table)

            cat_a, comp_a = tag_a.split(":", 1) if ":" in tag_a else ("Unknown", tag_a)
            cat_b, comp_b = tag_b.split(":", 1) if ":" in tag_b else ("Unknown", tag_b)

            rows.append({
                "Model": model_key,
                "Model_Display": MODEL_DISPLAY.get(model_key, model_key),
                "Category_1": cat_a,
                "Component_1": comp_a,
                "Category_2": cat_b,
                "Component_2": comp_b,
                "Modules": f"{cat_a} - {cat_b}",
                "Components": f"{comp_a} + {comp_b}",
                "Pair": f"{tag_a} + {tag_b}",
                "N": int(both.sum()),
                "Obs": obs,
                "Exp": exp,
                "Delta": (obs - exp) * 100,  # Negative indicates lower refusal than expected
                "Synergy_Score": score,
                "p_val": pval,
            })

    res = pd.DataFrame(rows)
    # FDR per model
    fdr_rows = []
    for _, mdf in res.groupby("Model"):
        mdf = mdf.copy()
        if len(mdf) > 0:
            _, qvals, _, _ = multipletests(mdf["p_val"], method="fdr_bh")
            mdf["q_val"] = qvals
        fdr_rows.append(mdf)

    final_df = pd.concat(fdr_rows, ignore_index=True) if fdr_rows else pd.DataFrame()
    _CACHE[cache_key] = final_df
    try:
        final_df.to_json(cache_file, orient="records", indent=2)
    except Exception:
        pass
    return final_df


def get_table4_pairwise_synergy(
    model_name: str = "All",
    min_cooccur: int = 10,
    min_solo: int = 10,
    top_k: int = 3
) -> pd.DataFrame:
    """Compute Table 4: Top component pairs per model by pairwise synergy."""
    df_syn = _compute_pairwise_synergy_cache(min_cooccur=min_cooccur, min_solo=min_solo)
    if df_syn.empty:
        return pd.DataFrame()

    sub = df_syn.copy()
    if model_name != "All":
        sub = sub[sub["Model_Display"] == model_name]

    rows = []
    for m, mdf in sub.groupby("Model_Display"):
        top_pairs = mdf.sort_values("Delta", ascending=True).head(top_k)
        for _, r in top_pairs.iterrows():
            q_str = "< 0.001" if r["q_val"] < 0.001 else ("< 0.01" if r["q_val"] < 0.01 else ("< 0.05" if r["q_val"] < 0.05 else f"{r['q_val']:.3f}"))
            rows.append({
                "Model": m,
                "Modules": r["Modules"],
                "Components": r["Components"],
                "Expected (%)": round(r["Exp"] * 100, 1),
                "Observed (%)": round(r["Obs"] * 100, 1),
                "Delta (pp)": round(r["Delta"], 1),
                "FDR q": q_str,
                "Co-occurrences (n)": r["N"],
            })

    return pd.DataFrame(rows)


def plot_pairwise_synergy_chart(model_name: str, top_k: int = 3) -> plt.Figure:
    """Plot Expected vs Observed refusal comparison for top synergistic pairs."""
    df_top = get_table4_pairwise_synergy(model_name=model_name, top_k=top_k)
    if df_top.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No synergy pairs available.", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(df_top) * 0.9)), dpi=120)
    y = np.arange(len(df_top))
    bar_height = 0.35

    ax.barh(y - bar_height / 2, df_top["Expected (%)"], height=bar_height, label="Expected Refusal (Independent)", color="#90CAF9")
    ax.barh(y + bar_height / 2, df_top["Observed (%)"], height=bar_height, label="Observed Refusal (Joint)", color="#D32F2F")

    ax.set_yticks(y)
    ax.set_yticklabels(df_top["Components"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Refusal Rate (%)", fontsize=10.5)
    ax.set_xlim(0, 115)
    ax.set_title(f"Pairwise Synergy Analysis: {model_name} (Top Combinations)", fontsize=11.5, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    for i, (_, row) in enumerate(df_top.iterrows()):
        ax.text(row["Observed (%)"] + 1.5, i + bar_height / 2,
                f"Obs: {row['Observed (%)']}% (Δ: {row['Delta (pp)']}pp, q={row['FDR q']})",
                va="center", fontsize=8.5, fontweight="bold", color="#B71C1C")

    fig.tight_layout()
    return fig


def plot_structural_density_curve() -> plt.Figure:
    """Plot Refusal Rate vs Structural Density (# components in prompt)."""
    df = _get_eval_df()
    cap = 12
    sub = df.copy()
    sub["k"] = sub["Num_Components"].clip(upper=cap).astype(int)

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)
    for m in MODEL_ORDER:
        mdf = sub[sub["Model"] == m]
        if mdf.empty:
            continue
        g = mdf.groupby("k")["Refusal"].mean().mul(100)
        m_disp = MODEL_DISPLAY.get(m, m)
        color = MODEL_COLORS.get(m_disp, "#4C78A8")
        ax.plot(g.index, g.values, marker="o", linewidth=1.8, markersize=5, label=m_disp, color=color)

    ax.set_xlabel("Structural Density (Total Components in Prompt)", fontsize=11)
    ax.set_ylabel("Refusal Rate (%)", fontsize=11)
    ax.set_title("Vulnerability Degradation by Structural Density (Prompt Complexity)", fontsize=12, pad=10)
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(1, cap + 1))
    ax.set_xticklabels([str(i) for i in range(1, cap)] + [f"{cap}+"])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout()
    return fig


# Model-specific sensitivity fingerprints

def _build_fingerprint_matrix(view: str) -> Tuple[pd.DataFrame, List[str]]:
    """Build fingerprint vectors across models for the selected view."""
    labels = [MODEL_DISPLAY[m] for m in MODEL_ORDER]

    if view.startswith("Single"):
        all_ranks = compute_single_module_ranking_per_model(min_support=10)
        all_tags = sorted((all_ranks["Category"] + ":" + all_ranks["Component"]).unique())

        vectors = {}
        for m in MODEL_ORDER:
            mdf = all_ranks[all_ranks["Model"] == m].copy()
            mdf["Tag"] = mdf["Category"] + ":" + mdf["Component"]
            # Refusal_Diff = Baseline - Mean (positive indicates refusal dropped)
            s = (-mdf.set_index("Tag")["Delta_pp"] / 100).reindex(all_tags, fill_value=0.0)
            vectors[m] = s

        mat = pd.DataFrame(vectors).T
        mat.index = labels

        if "Drop" in view:
            return mat.clip(lower=0), labels
        elif "Rise" in view:
            return (-mat).clip(lower=0), labels
        else:
            return mat, labels

    else:
        # Pairwise synergy
        df_syn = _compute_pairwise_synergy_cache(min_cooccur=10, min_solo=10)
        all_pairs = sorted(df_syn["Pair"].unique())

        vectors = {}
        for m in MODEL_ORDER:
            mdf = df_syn[df_syn["Model"] == m]
            if not mdf.empty:
                s = mdf.set_index("Pair")["Synergy_Score"].reindex(all_pairs, fill_value=0.0)
            else:
                s = pd.Series(0.0, index=all_pairs)
            vectors[m] = s

        mat = pd.DataFrame(vectors).T
        mat.index = labels

        if "Drop" in view:
            return mat.clip(lower=0), labels
        elif "Rise" in view:
            return (-mat).clip(lower=0), labels
        else:
            return mat, labels


def get_table7_cosine_similarity(view: str = "Pairwise: Integrated (Full Synergy)") -> pd.DataFrame:
    """Compute 6x6 Cosine Similarity Matrix matching Table 7 & Figure 2a."""
    mat, labels = _build_fingerprint_matrix(view)
    n = len(labels)
    sim = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                sim[i, j] = 1.0
            else:
                v1, v2 = mat.iloc[i].values, mat.iloc[j].values
                if np.all(v1 == 0) or np.all(v2 == 0):
                    sim[i, j] = 0.0
                else:
                    sim[i, j] = float(1.0 - cosine(v1, v2))

    df_sim = pd.DataFrame(sim, index=labels, columns=labels)
    return df_sim.round(3)


def get_all_15_pairs_similarity_table() -> pd.DataFrame:
    """Generate complete Table 7 showing all 15 model pairs across 6 fingerprint views."""
    views = [
        ("Single Drop", "Single: Drop (Vulnerability)"),
        ("Single Rise", "Single: Rise (Over-refusal)"),
        ("Single Integ.", "Single: Integrated (Full Structure)"),
        ("Pairwise Drop", "Pairwise: Drop (Synergy Vulnerability)"),
        ("Pairwise Rise", "Pairwise: Rise (Synergy Over-refusal)"),
        ("Pairwise Integ.", "Pairwise: Integrated (Full Synergy)"),
    ]

    mats = {name: get_table7_cosine_similarity(v) for name, v in views}
    labels = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    rows = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            m1, m2 = labels[i], labels[j]
            rows.append({
                "Model Pair": f"{m1} - {m2}",
                "Single Drop": mats["Single Drop"].loc[m1, m2],
                "Single Rise": mats["Single Rise"].loc[m1, m2],
                "Single Integ.": mats["Single Integ."].loc[m1, m2],
                "Pairwise Drop": mats["Pairwise Drop"].loc[m1, m2],
                "Pairwise Rise": mats["Pairwise Rise"].loc[m1, m2],
                "Pairwise Integ.": mats["Pairwise Integ."].loc[m1, m2],
            })

    return pd.DataFrame(rows)


def plot_fingerprint_combined_figure(view: str = "Pairwise: Integrated (Full Synergy)") -> plt.Figure:
    """Generate cosine similarity matrix and hierarchical clustering dendrogram."""
    df_sim = get_table7_cosine_similarity(view)
    labels = list(df_sim.index)

    dist = 1.0 - df_sim.values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist), method="ward")

    fig = plt.figure(figsize=(14.5, 5.2), dpi=120)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.32)

    # Similarity Heatmap
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(df_sim.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_title("Cosine Similarity Matrix", fontsize=11, pad=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = df_sim.values[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax1.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=9)

    # Dendrogram with right-aligned leaf labels
    ax2 = fig.add_subplot(gs[1])
    dendrogram(
        Z, labels=labels, ax=ax2, orientation="left",
        color_threshold=0, above_threshold_color="#555555"
    )
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel("Ward Distance (1 - Cosine Similarity)", fontsize=10)
    ax2.set_title("Hierarchical Clustering (Ward Linkage)", fontsize=11, pad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    for tick in ax2.get_yticklabels():
        m_text = tick.get_text()
        m_orig = next((k for k, v in MODEL_DISPLAY.items() if v == m_text), None)
        t_type = MODEL_TYPE.get(m_orig, "Closed")
        tick.set_color(TYPE_COLORS.get(t_type, "black"))
        tick.set_fontweight("bold")
        tick.set_fontsize(9)

    legend_elements = [
        Patch(facecolor=TYPE_COLORS["Closed"], label="Closed-weight (Proprietary)"),
        Patch(facecolor=TYPE_COLORS["Open"], label="Open-weight (Public)")
    ]
    ax2.legend(handles=legend_elements, loc="upper left", fontsize=8.5)

    fig.suptitle(f"Model Alignment Sensitivity Fingerprinting: {view}", fontsize=12.5, y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.20, left=0.12, right=0.86, wspace=0.32)
    return fig


def get_model_pair_comparison(
    model_a: str,
    model_b: str,
    view: str = "Single: Integrated (Full Structure)"
) -> Tuple[float, pd.DataFrame, plt.Figure]:
    """Compare two selected models: shared vulnerabilities vs unique sensitivities."""
    mat, labels = _build_fingerprint_matrix(view)
    if model_a not in mat.index or model_b not in mat.index:
        empty_fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        return 0.0, pd.DataFrame(), empty_fig

    v1 = mat.loc[model_a].values
    v2 = mat.loc[model_b].values

    sim = float(1.0 - cosine(v1, v2)) if not (np.all(v1 == 0) or np.all(v2 == 0)) else 0.0

    features = mat.columns
    df_diff = pd.DataFrame({
        "Feature": features,
        model_a: v1,
        model_b: v2,
        "Abs_Difference": np.abs(v1 - v2),
        "Mean_Impact": (v1 + v2) / 2,
    }).sort_values("Mean_Impact", ascending=False)

    top_feats = df_diff.head(10).copy()

    fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
    y = np.arange(len(top_feats))
    bh = 0.35

    ax.barh(y - bh / 2, top_feats[model_a], height=bh, label=model_a, color=MODEL_COLORS.get(model_a, "#D32F2F"))
    ax.barh(y + bh / 2, top_feats[model_b], height=bh, label=model_b, color=MODEL_COLORS.get(model_b, "#1976D2"))

    ax.set_yticks(y)
    ax.set_yticklabels(top_feats["Feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Vulnerability Lift / Synergy Score", fontsize=10.5)
    ax.set_title(f"Head-to-Head Structural Sensitivity: {model_a} vs {model_b} (Cosine Sim: {sim:.3f})", fontsize=11.5, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    return round(sim, 3), df_diff.head(20).round(3), fig
