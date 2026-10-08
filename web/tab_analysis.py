# web/tab_analysis.py
from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.data_utils import (
    get_model_choices,
    get_dataset_choices,
    truncate_text,
    load_and_process_eval_data,
)

# Config
TYPE_MAP = {
    "roles": "Role",
    "domains": "Domain",
    "actions": "Action",
    "objects": "Object",
    "formats": "Format",
}
MODULE_KEYS = list(TYPE_MAP.keys())
MODULE_LABELS = list(TYPE_MAP.values())

BAYES_M = 5.0
DEFAULT_MIN_COUNT = 10
PAIR_MIN_COUNT = 5
FIG_H = 420
JB_THRESHOLD = 0.5

# Publication-quality visual theme (shared with tab_comparative_analysis.py)
FONT_FAMILY = "Arial, 'Helvetica Neue', Helvetica, sans-serif"
BG_COLOR = "#FFFFFF"
GRID_COLOR = "#E9E9E9"
TEXT_COLOR = "#1A1A1A"
MUTED_TEXT = "#6B6B6B"

COLOR_RESPONSE = "#D62728"   # jailbreak / danger
COLOR_REFUSAL = "#2CA02C"    # safe / refusal

# Colorblind-safe categorical palette (Okabe-Ito)
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C564B", "#7F7F7F"]


# Generic helpers
def EMPTY_FIG(title: str = "No Data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"️  {title}",
        showarrow=False,
        font=dict(size=16, color=MUTED_TEXT, family=FONT_FAMILY),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    fig.update_layout(
        template="plotly_white",
        height=FIG_H,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    return fig


def _style(
    fig: go.Figure,
    *,
    height: Optional[int] = None,
    margin: Optional[dict] = None,
    show_coloraxis: bool = False,
    **kwargs,
) -> go.Figure:
    """Apply the shared publication theme to a figure."""
    base = dict(
        template="plotly_white",
        height=height or FIG_H,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        coloraxis_showscale=show_coloraxis,
        title_x=0.02,
        font=dict(family=FONT_FAMILY, size=13, color=TEXT_COLOR),
        title_font=dict(size=16.5, family=FONT_FAMILY, color="#0D0D0D"),
        legend=dict(
            font=dict(size=12, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DDDDDD",
            borderwidth=1,
        ),
        margin=margin or dict(t=60, b=40, l=50, r=30),
        hoverlabel=dict(font_size=13, font_family=FONT_FAMILY, bgcolor="white"),
    )
    base.update(kwargs)
    fig.update_layout(**base)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                      title_font=dict(size=13, family=FONT_FAMILY), tickfont=dict(size=11.5))
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                      title_font=dict(size=13, family=FONT_FAMILY), tickfont=dict(size=11.5))
    return fig


def _safe_float(value, default=0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _bayes_rate(count: int, successes: int, global_rate: float, m: float = BAYES_M) -> float:
    if count <= 0:
        return global_rate
    return (count * (successes / count) + m * global_rate) / (count + m)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score confidence interval for a proportion k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2)))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _filter_df(df: Optional[pd.DataFrame], selected_categories) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    if selected_categories and "Category" in df.columns:
        return df[df["Category"].isin(selected_categories)].copy()
    return df.copy()


def _global_rate(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Is_Jailbroken" not in df.columns:
        return 0.0
    return float(df["Is_Jailbroken"].mean())


# Data loading
def load_data(selected_model, selected_datasets):
    if not selected_datasets:
        return None, []

    dfs = []
    for dataset in selected_datasets:
        df = load_and_process_eval_data(selected_model, dataset)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return None, []

    combined = pd.concat(dfs, ignore_index=True)
    if "Prompt" in combined.columns:
        combined = combined.drop_duplicates(subset=["Prompt"], keep="first")

    categories = (
        sorted(combined["Category"].dropna().astype(str).unique().tolist())
        if "Category" in combined.columns
        else []
    )
    return combined, categories


# PENTA parsing / materialization
def _parse_raw_categories(raw) -> Dict[str, List[str]]:
    empty = {k: [] for k in MODULE_KEYS}
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return empty
    try:
        cats = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(cats, dict):
            return empty
        out = {}
        for key in MODULE_KEYS:
            items = cats.get(key, cats.get(key.capitalize(), []))
            if isinstance(items, list):
                seen, cleaned = set(), []
                for item in items:
                    if item is None:
                        continue
                    text = str(item).strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    cleaned.append(text)
                out[key] = cleaned
            elif items:
                out[key] = [str(items).strip()]
            else:
                out[key] = []
        return out
    except Exception:
        return empty


def _attach_penta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-shot materialization used by all charts.

    Adds:
      Role/Domain/Action/Object/Format (lists)
      Total_Components, Present_Modules, Configuration
      Component_Labels  ([Type] item unique list per prompt)
      Is_Jailbroken, Response_Type
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    parsed = result["raw_categories"].map(_parse_raw_categories)

    for key, label in TYPE_MAP.items():
        result[label] = parsed.map(lambda x, k=key: x.get(k, []))

    result["Total_Components"] = parsed.map(
        lambda x: sum(len(x.get(k, [])) for k in MODULE_KEYS)
    )
    result["Present_Modules"] = parsed.map(
        lambda x: [TYPE_MAP[k] for k in MODULE_KEYS if x.get(k)]
    )
    result["Configuration"] = result["Present_Modules"].map(
        lambda mods: " + ".join(mods) if mods else "Core (Unstructured)"
    )

    # Precompute "[Role] x" labels once — reused by component / pair / domain charts
    def _labels(x: dict) -> List[str]:
        out, seen = [], set()
        for k, lab in TYPE_MAP.items():
            for item in x.get(k, []):
                s = f"[{lab}] {item}"
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out

    result["Component_Labels"] = parsed.map(_labels)

    # Vectorized binary label
    asr = pd.to_numeric(result.get("ASR", 0), errors="coerce").fillna(0.0)
    result["Is_Jailbroken"] = (asr > JB_THRESHOLD).astype(np.int8)
    result["Response_Type"] = np.where(result["Is_Jailbroken"] == 1, "Response", "Refusal")
    return result


# Shared group stats
def _group_statistics(df: pd.DataFrame, group_col: str, min_count: int = 1) -> pd.DataFrame:
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame()

    g = df.groupby(group_col, sort=False)
    n = g.size()
    jb = g["Is_Jailbroken"].sum()
    mean_score = g["ASR"].mean() if "ASR" in df.columns else pd.Series(0.0, index=n.index)
    global_rate = _global_rate(df)

    out = pd.DataFrame(
        {
            group_col: n.index,
            "N": n.values,
            "Jailbreaks": jb.values.astype(int),
            "Mean_Score": mean_score.values,
        }
    )
    out = out[out["N"] >= min_count]
    if out.empty:
        return out

    out["Jailbreak_Rate"] = out["Jailbreaks"] / out["N"]
    out["Smoothed_Rate"] = out.apply(
        lambda r: _bayes_rate(int(r["N"]), int(r["Jailbreaks"]), global_rate),
        axis=1,
    )
    out["Delta_vs_Baseline"] = out["Jailbreak_Rate"] - global_rate
    ci = out.apply(lambda r: _wilson_ci(int(r["Jailbreaks"]), int(r["N"])), axis=1)
    out["CI_Lo"], out["CI_Hi"] = zip(*ci)
    return out.reset_index(drop=True)


def _count_labeled_events(
    df: pd.DataFrame,
    label_lists: pd.Series,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Generic one-pass counter for list-valued labels.
    label_lists: Series of List[str]
    """
    global_rate = _global_rate(df)
    stats: Dict[str, List[int]] = {}  # label -> [n, jb]

    jb_arr = df["Is_Jailbroken"].to_numpy()
    for i, labels in enumerate(label_lists):
        if not labels:
            continue
        is_jb = int(jb_arr[i])
        for lab in labels:
            st = stats.get(lab)
            if st is None:
                stats[lab] = [1, is_jb]
            else:
                st[0] += 1
                st[1] += is_jb

    rows = []
    for lab, (n, jb) in stats.items():
        if n < min_count:
            continue
        raw = jb / n
        lo, hi = _wilson_ci(jb, n)
        rows.append(
            {
                "Label": lab,
                "N": n,
                "Jailbreaks": jb,
                "Jailbreak_Rate": raw,
                "Smoothed_Rate": _bayes_rate(n, jb, global_rate),
                "Delta_vs_Baseline": raw - global_rate,
                "CI_Lo": lo,
                "CI_Hi": hi,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# KPI
def generate_kpi_text(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "️ No data available for the selected filters."

    n = len(df)
    jailbreaks = int(df["Is_Jailbroken"].sum())
    rate = jailbreaks / n if n else 0.0
    lo, hi = _wilson_ci(jailbreaks, n)
    mean_score = float(df["ASR"].mean()) if "ASR" in df.columns else 0.0
    n_categories = df["Category"].nunique() if "Category" in df.columns else 0
    avg_components = float(df["Total_Components"].mean()) if "Total_Components" in df.columns else 0.0

    # NOTE: every line below is flush-left (no leading whitespace). gr.Markdown
    # treats 4+ leading spaces on a line as a fenced code block, which would
    # cause this HTML to render as raw text instead of a styled card.
    def stat_block(label, value, sub="", color=TEXT_COLOR):
        return (
            f'<div style="flex:1; min-width:160px; padding:14px 18px; background:#ffffff; '
            f'border:1px solid #e6e6e6; border-radius:10px; box-shadow:0 1px 2px rgba(0,0,0,0.03);">'
            f'<div style="font-size:11.5px; color:#8a8a8a; text-transform:uppercase; letter-spacing:.05em; font-weight:600;">{label}</div>'
            f'<div style="font-size:26px; font-weight:700; color:{color}; margin-top:3px; font-family:{FONT_FAMILY};">{value}</div>'
            f'<div style="font-size:12px; color:#9a9a9a; margin-top:3px;">{sub}</div>'
            f'</div>'
        )

    blocks = "".join([
        stat_block("Jailbreak Rate", f"{rate*100:.1f}%", f"95% CI {lo*100:.1f}–{hi*100:.1f}%", COLOR_RESPONSE),
        stat_block("Refusal Rate", f"{(1-rate)*100:.1f}%", f"{n - jailbreaks:,} of {n:,} prompts", COLOR_REFUSAL),
        stat_block("Mean Evaluator Score", f"{mean_score:.3f}", "average ASR score"),
        stat_block("Avg. PENTA Components", f"{avg_components:.2f}", f"across {n_categories} categories"),
    ])

    return (
        f'<div style="padding:18px 20px; border-radius:14px; background:#f7f7f8; border:1px solid #e5e5e5; font-family:{FONT_FAMILY};">'
        f'<h3 style="margin:0 0 3px 0; color:#111; font-size:18px;"> Evaluation Overview</h3>'
        f'<div style="color:#777; font-size:13px; margin-bottom:14px;">{n:,} total prompts evaluated</div>'
        f'<div style="display:flex; gap:14px; flex-wrap:wrap;">{blocks}</div>'
        f'</div>'
    )


# 1. Baseline
def fig_overall_response_pie(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return EMPTY_FIG("Overall Jailbreak / Refusal Rate")

    counts = (
        df["Response_Type"]
        .map({"Response": "Jailbroken ", "Refusal": "Refused ️"})
        .value_counts()
        .rename_axis("Response_Type")
        .reset_index(name="Count")
    )
    fig = px.pie(
        counts,
        names="Response_Type",
        values="Count",
        hole=0.5,
        title=" Overall Jailbreak / Refusal Rate",
        color="Response_Type",
        color_discrete_map={"Jailbroken ": COLOR_RESPONSE, "Refused ️": COLOR_REFUSAL},
    )
    fig.update_traces(
        textposition="inside", textinfo="percent+label",
        textfont=dict(size=13, color="white"),
        marker=dict(line=dict(color="white", width=2)),
    )
    return _style(fig, showlegend=False, margin=dict(t=60, b=20, l=20, r=20))


def fig_category_vulnerability(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty or "Category" not in df.columns:
        return EMPTY_FIG("Category-level Jailbreak Association")

    stats = _group_statistics(df, "Category", min_count=3)
    if stats.empty:
        return EMPTY_FIG("Not enough category support")

    stats = stats.sort_values("Jailbreak_Rate", ascending=True)
    stats["ErrPlus"] = stats["CI_Hi"] - stats["Jailbreak_Rate"]
    stats["ErrMinus"] = stats["Jailbreak_Rate"] - stats["CI_Lo"]

    fig = px.bar(
        stats,
        x="Jailbreak_Rate",
        y="Category",
        orientation="h",
        color="Jailbreak_Rate",
        color_continuous_scale="Reds",
        text="Jailbreak_Rate",
        error_x="ErrPlus",
        error_x_minus="ErrMinus",
        title=" Jailbreak Rate by Threat Category (error bars = 95% Wilson CI)",
        hover_data=["N", "Jailbreaks", "Mean_Score", "Delta_vs_Baseline"],
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", textfont=dict(size=11))
    return _style(
        fig,
        height=max(FIG_H, 32 * len(stats) + 110),
        margin=dict(t=60, b=40, l=190, r=40),
        xaxis_title="Jailbreak Rate",
        yaxis_title="",
        xaxis_tickformat=".0%",
    )


# 2. Component density
def fig_component_counts_by_response(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return EMPTY_FIG("Component Density by Response Type")

    # Vectorized length columns
    lengths = {lab: df[lab].map(len) for lab in MODULE_LABELS}
    lengths["Total"] = df["Total_Components"]
    tmp = pd.DataFrame(lengths)
    tmp["Response_Type"] = df["Response_Type"].values

    agg = tmp.groupby("Response_Type")[MODULE_LABELS + ["Total"]].mean().reset_index()
    melted = agg.melt(id_vars="Response_Type", var_name="Module", value_name="Mean_Count")

    fig = px.bar(
        melted,
        x="Module",
        y="Mean_Count",
        color="Response_Type",
        barmode="group",
        color_discrete_map={"Response": COLOR_RESPONSE, "Refusal": COLOR_REFUSAL},
        text_auto=".2f",
        title=" Mean PENTA Component Density",
    )
    fig.update_traces(textfont=dict(size=11), marker_line=dict(width=0.5, color="white"))
    return _style(
        fig,
        yaxis_title="Mean number of components",
        xaxis_title="",
        legend=dict(orientation="h", y=1.1, font=dict(size=12)),
        margin=dict(t=70, b=40, l=50, r=20),
    )


def fig_total_density_by_category(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty or "Category" not in df.columns:
        return EMPTY_FIG("Component Density by Category")

    stats = (
        df.groupby(["Category", "Response_Type"], sort=False)["Total_Components"]
        .mean()
        .reset_index()
    )
    if stats.empty:
        return EMPTY_FIG("Component Density by Category")

    fig = px.bar(
        stats,
        x="Category",
        y="Total_Components",
        color="Response_Type",
        barmode="group",
        color_discrete_map={"Response": COLOR_RESPONSE, "Refusal": COLOR_REFUSAL},
        text_auto=".2f",
        title=" Mean Total PENTA Components · Category × Outcome",
    )
    fig.update_traces(textfont=dict(size=10.5), marker_line=dict(width=0.5, color="white"))
    return _style(
        fig,
        xaxis_tickangle=-30,
        yaxis_title="Mean total components",
        xaxis_title="",
        legend=dict(orientation="h", y=1.1, font=dict(size=12)),
        margin=dict(t=70, b=90, l=50, r=20),
    )


def fig_component_density_heatmap(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty or "Category" not in df.columns:
        return EMPTY_FIG("Component Density Heatmap")

    resp = df[df["Response_Type"] == "Response"]
    if resp.empty:
        return EMPTY_FIG("Response-only Component Density")

    # mean list-length per module without applymap
    data = {lab: resp[lab].map(len) for lab in MODULE_LABELS}
    matrix = pd.DataFrame(data)
    matrix["Category"] = resp["Category"].values
    matrix = matrix.groupby("Category", sort=False)[MODULE_LABELS].mean()

    if matrix.empty:
        return EMPTY_FIG("Response-only Component Density")

    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title=" Response-only Mean Component Density by Threat Category",
        labels=dict(color="Mean count"),
    )
    fig.update_traces(textfont=dict(size=11))
    fig.update_coloraxes(colorbar=dict(title="Mean count"))
    return _style(fig, height=max(FIG_H, 30 * len(matrix) + 130), margin=dict(t=60, b=40, l=140, r=30), show_coloraxis=True)


# 3. Structural configuration
def fig_structural_configuration(df: pd.DataFrame, min_count: int = 5) -> go.Figure:
    if df is None or df.empty:
        return EMPTY_FIG("Jailbreak Rate by Structural Configuration")

    stats = _group_statistics(df, "Configuration", min_count=min_count)
    if stats.empty:
        return EMPTY_FIG("Not enough structural configuration support")

    stats = stats.sort_values("Smoothed_Rate", ascending=True)
    fig = px.bar(
        stats,
        x="Smoothed_Rate",
        y="Configuration",
        orientation="h",
        color="Smoothed_Rate",
        color_continuous_scale="Plasma",
        text="Smoothed_Rate",
        title=" Jailbreak Association by PENTA Structural Configuration (Bayesian-smoothed)",
        hover_data=["N", "Jailbreaks", "Jailbreak_Rate", "Mean_Score", "Delta_vs_Baseline", "CI_Lo", "CI_Hi"],
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="inside", textfont=dict(size=11, color="white"))
    return _style(
        fig,
        height=max(FIG_H, 38 * len(stats) + 110),
        margin=dict(l=260, r=40, t=60, b=40),
        xaxis_title="Smoothed Jailbreak Rate",
        yaxis_title="",
        xaxis_tickformat=".0%",
    )


# 4. Individual components
def fig_top_individual_components(
    df: pd.DataFrame, min_count: int = 5, top_k: int = 12
) -> go.Figure:
    if df is None or df.empty:
        return EMPTY_FIG("No component statistics available")

    stats = _count_labeled_events(df, df["Component_Labels"], min_count=min_count)
    if stats.empty:
        return EMPTY_FIG("No components meet minimum support")

    stats = (
        stats.sort_values(["Smoothed_Rate", "N"], ascending=[False, False])
        .head(top_k)
        .sort_values("Smoothed_Rate", ascending=True)
    )
    global_rate = _global_rate(df)

    fig = px.bar(
        stats,
        x="Smoothed_Rate",
        y="Label",
        orientation="h",
        color="Smoothed_Rate",
        color_continuous_scale="YlOrRd",
        text="Smoothed_Rate",
        title=f" Top {top_k} Components by Jailbreak Association (Bayesian-smoothed)",
        hover_data=["N", "Jailbreaks", "Jailbreak_Rate", "Delta_vs_Baseline", "CI_Lo", "CI_Hi"],
    )
    fig.add_vline(
        x=global_rate,
        line_dash="dash",
        line_color=COLOR_RESPONSE,
        line_width=1.5,
        annotation_text=f"Baseline ({global_rate:.1%})",
        annotation_position="top left",
        annotation_font=dict(size=11, color=COLOR_RESPONSE),
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="inside", textfont=dict(size=11, color="white"))
    return _style(
        fig,
        height=max(FIG_H, 34 * len(stats) + 110),
        margin=dict(l=270, r=40, t=60, b=40),
        xaxis_title="Smoothed Jailbreak Rate",
        yaxis_title="",
        xaxis_tickformat=".0%",
    )


# 5. Domain-centric
def fig_domain_response_ratio(df: pd.DataFrame, min_count: int = 5) -> go.Figure:
    if df is None or df.empty:
        return EMPTY_FIG("Domain-level Jailbreak Association")

    # explode domains without Python row loop where possible
    tmp = df[["Domain", "Is_Jailbroken", "ASR"]].copy()
    tmp = tmp.explode("Domain")
    tmp["Domain"] = tmp["Domain"].fillna("(No Domain)")
    tmp.loc[tmp["Domain"].astype(str).str.len() == 0, "Domain"] = "(No Domain)"

    stats = (
        tmp.groupby("Domain", sort=False)
        .agg(N=("Is_Jailbroken", "size"), Jailbreak_Rate=("Is_Jailbroken", "mean"), Mean_Score=("ASR", "mean"))
        .reset_index()
    )
    stats = stats[stats["N"] >= min_count]
    if stats.empty:
        return EMPTY_FIG(f"No domains with N ≥ {min_count}")

    stats = stats.sort_values("Jailbreak_Rate", ascending=True)
    stats["Refusal_Rate"] = 1.0 - stats["Jailbreak_Rate"]

    fig = go.Figure()
    fig.add_bar(
        name="Refusal",
        y=stats["Domain"],
        x=stats["Refusal_Rate"],
        orientation="h",
        marker_color=COLOR_REFUSAL,
        hovertemplate="%{y}<br>Refusal: %{x:.1%}<extra></extra>",
    )
    fig.add_bar(
        name="Jailbreak",
        y=stats["Domain"],
        x=stats["Jailbreak_Rate"],
        orientation="h",
        marker_color=COLOR_RESPONSE,
        hovertemplate="%{y}<br>Jailbreak: %{x:.1%}<extra></extra>",
    )
    return _style(
        fig,
        barmode="stack",
        title=" Domain-wise Jailbreak / Refusal Rate",
        xaxis_title="Rate",
        yaxis_title="",
        xaxis_tickformat=".0%",
        height=max(FIG_H, 30 * len(stats) + 130),
        margin=dict(l=190, r=30, t=70, b=40),
        legend=dict(orientation="h", y=1.08, font=dict(size=12)),
    )


def fig_domain_top_component(df: pd.DataFrame, min_count: int = 10) -> go.Figure:
    """Domain × single highest-association non-Domain component."""
    if df is None or df.empty:
        return EMPTY_FIG("Domain × High-Association Component")

    global_rate = _global_rate(df)
    # domain -> component -> [n, jb]
    stats: Dict[str, Dict[str, List[int]]] = {}

    jb = df["Is_Jailbroken"].to_numpy()
    domains_col = df["Domain"].tolist()
    labels_col = df["Component_Labels"].tolist()

    for i in range(len(df)):
        domains = domains_col[i] or []
        if not domains:
            continue
        is_jb = int(jb[i])
        # strip domain-typed labels
        others = [lab for lab in labels_col[i] if not lab.startswith("[Domain] ")]
        if not others:
            continue
        for d in domains:
            bucket = stats.setdefault(d, {})
            for lab in others:
                st = bucket.get(lab)
                if st is None:
                    bucket[lab] = [1, is_jb]
                else:
                    st[0] += 1
                    st[1] += is_jb

    rows = []
    for domain, bucket in stats.items():
        best = None
        for lab, (n, jbv) in bucket.items():
            if n < min_count:
                continue
            raw = jbv / n
            sm = _bayes_rate(n, jbv, global_rate)
            cand = (sm, n, lab, raw, jbv)
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best is None:
            continue
        sm, n, lab, raw, jbv = best
        rows.append(
            {
                "Domain": domain,
                "Component": lab,
                "Label": f"{domain} × {lab}",
                "N": n,
                "Jailbreaks": jbv,
                "Jailbreak_Rate": raw,
                "Smoothed_Rate": sm,
                "Delta_vs_Baseline": raw - global_rate,
            }
        )

    if not rows:
        return EMPTY_FIG(f"No Domain × Component with N ≥ {min_count}")

    result = pd.DataFrame(rows).sort_values("Smoothed_Rate", ascending=True)
    fig = px.bar(
        result,
        x="Smoothed_Rate",
        y="Label",
        orientation="h",
        color="Smoothed_Rate",
        color_continuous_scale="YlOrRd",
        text="Smoothed_Rate",
        title=" Domain × Highest-Association Component (Bayesian-smoothed)",
        hover_data=["Domain", "Component", "N", "Jailbreaks", "Jailbreak_Rate", "Delta_vs_Baseline"],
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="inside", textfont=dict(size=10.5, color="white"))
    return _style(
        fig,
        height=max(FIG_H, 32 * len(result) + 130),
        margin=dict(l=320, r=40, t=60, b=40),
        xaxis_title="Smoothed Jailbreak Rate",
        yaxis_title="",
        xaxis_tickformat=".0%",
    )


# 6. Pairwise
def get_pairwise_data(df: pd.DataFrame, min_count: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame([{"Message": "No data."}])
    if df is None or df.empty:
        return empty, empty

    global_rate = _global_rate(df)
    type_stats: Dict[str, List[int]] = {}
    comp_stats: Dict[str, List[int]] = {}

    jb = df["Is_Jailbroken"].to_numpy()
    present_col = df["Present_Modules"].tolist()
    labels_col = df["Component_Labels"].tolist()

    for i in range(len(df)):
        is_jb = int(jb[i])

        present = sorted(set(present_col[i]))
        for a, b in combinations(present, 2):
            pair = f"{a} + {b}"
            st = type_stats.get(pair)
            if st is None:
                type_stats[pair] = [1, is_jb]
            else:
                st[0] += 1
                st[1] += is_jb

        # cross-module component pairs via label prefix [Module]
        labs = labels_col[i]
        for a, b in combinations(labs, 2):
            # module encoded in "[Module] ..."
            ma = a[1 : a.index("]")] if a.startswith("[") and "]" in a else ""
            mb = b[1 : b.index("]")] if b.startswith("[") and "]" in b else ""
            if not ma or ma == mb:
                continue
            x, y = (a, b) if a < b else (b, a)
            pair = f"{x} + {y}"
            st = comp_stats.get(pair)
            if st is None:
                comp_stats[pair] = [1, is_jb]
            else:
                st[0] += 1
                st[1] += is_jb

    def _to_table(stats: dict, name_col: str) -> pd.DataFrame:
        rows = []
        for pair, (n, jbv) in stats.items():
            if n < min_count:
                continue
            raw = jbv / n
            lo, hi = _wilson_ci(jbv, n)
            rows.append(
                {
                    name_col: pair,
                    "N": n,
                    "Jailbreaks": jbv,
                    "Jailbreak_Rate (%)": round(raw * 100, 1),
                    "Smoothed_Rate (%)": round(_bayes_rate(n, jbv, global_rate) * 100, 1),
                    "Delta_vs_Baseline (pp)": round((raw - global_rate) * 100, 1),
                    "95% CI (%)": f"{lo*100:.1f} – {hi*100:.1f}",
                }
            )
        if not rows:
            return pd.DataFrame([{"Message": "No pairs with enough support."}])
        return (
            pd.DataFrame(rows)
            .sort_values(["Smoothed_Rate (%)", "N"], ascending=[False, False])
            .head(10)
            .reset_index(drop=True)
        )

    return _to_table(type_stats, "Category Pair"), _to_table(comp_stats, "Component Pair")


# 7. Incremental module contribution
def get_incremental_contribution_data(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # presence sets
    present = df["Present_Modules"].tolist()
    jb = df["Is_Jailbroken"].to_numpy()

    # rate for exact configuration frozenset
    cfg_stats: Dict[frozenset, List[int]] = {}
    for i, mods in enumerate(present):
        key = frozenset(mods)
        st = cfg_stats.get(key)
        if st is None:
            cfg_stats[key] = [1, int(jb[i])]
        else:
            st[0] += 1
            st[1] += int(jb[i])

    rows = []
    modules = MODULE_LABELS
    for base in modules:
        for added in modules:
            if base == added:
                continue
            base_key = frozenset([base])
            comb_key = frozenset([base, added])
            if base_key not in cfg_stats or comb_key not in cfg_stats:
                continue
            bn, bj = cfg_stats[base_key]
            cn, cj = cfg_stats[comb_key]
            if bn < min_count or cn < min_count:
                continue
            br = bj / bn
            cr = cj / cn
            rows.append(
                {
                    "Base": base,
                    "Added": added,
                    "Base_N": bn,
                    "Base_Rate": br,
                    "Combined_N": cn,
                    "Combined_Rate": cr,
                    "Incremental_Delta": cr - br,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Incremental_Delta", ascending=False).reset_index(drop=True)


def fig_incremental_contribution(df: pd.DataFrame, min_count: int = 5) -> go.Figure:
    stats = get_incremental_contribution_data(df, min_count=min_count)
    if stats.empty:
        return EMPTY_FIG("No Conditional Module Comparisons")

    stats = stats.copy()
    stats["Transition"] = stats["Base"] + " → " + stats["Added"]
    stats = stats.sort_values("Incremental_Delta", ascending=True)

    fig = px.bar(
        stats,
        x="Incremental_Delta",
        y="Transition",
        orientation="h",
        color="Incremental_Delta",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        text="Incremental_Delta",
        title=" Incremental Module Contribution (Conditional Association)",
        hover_data=["Base_N", "Base_Rate", "Combined_N", "Combined_Rate"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color=MUTED_TEXT, line_width=1.2)
    fig.update_traces(texttemplate="%{text:+.1%}", textposition="outside", textfont=dict(size=10.5))
    fig.update_coloraxes(colorbar=dict(title="Δ Rate", tickformat="+.0%"))
    return _style(
        fig,
        height=max(FIG_H, 32 * len(stats) + 130),
        margin=dict(l=190, r=70, t=70, b=50),
        xaxis_title="Δ Jailbreak Rate (Base + Added − Base alone)",
        yaxis_title="",
        xaxis_tickformat="+.0%",
    )


# Pair inspection
def handle_pair_click(evt: gr.SelectData, pair_df, full_df):
    empty = pd.DataFrame([{"Message": "No data available."}])
    if full_df is None or full_df.empty or pair_df is None or pair_df.empty:
        return empty
    if "Message" in getattr(pair_df, "columns", []):
        return empty

    try:
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        pair_string = str(pair_df.iloc[row_idx].iloc[0])
        parts = [p.strip() for p in pair_string.split(" + ")]
    except Exception:
        return empty
    if len(parts) != 2:
        return empty

    is_module_pair = all(p in MODULE_LABELS for p in parts)
    matches = []

    jb = full_df["Is_Jailbroken"].to_numpy()
    for i, row in enumerate(full_df.itertuples(index=False)):
        if int(jb[i]) != 1:
            continue
        if is_module_pair:
            present = set(getattr(row, "Present_Modules"))
            ok = all(p in present for p in parts)
        else:
            labels = set(getattr(row, "Component_Labels"))
            ok = all(p in labels for p in parts)
        if ok:
            matches.append(
                {
                    "Prompt": truncate_text(getattr(row, "Prompt", ""), 180),
                    "Target Response": truncate_text(full_df.iloc[i].get("Target Response", ""), 220),
                    "Score": _safe_float(full_df.iloc[i].get("ASR", 0)),
                    "Category": full_df.iloc[i].get("Category", ""),
                }
            )

    if not matches:
        return pd.DataFrame(
            [{"Message": "No successful jailbreak prompts found for this pair."}]
        )
    return pd.DataFrame(matches)


def prep_table(df, selected_categories):
    if df is None or df.empty:
        return None
    result = _filter_df(df, selected_categories)
    priority = ["Index", "Category", "ASR", "Prompt", "Target Response"]
    cols = [c for c in priority if c in result.columns]
    return result[cols] if cols else result


# Master update
def on_update_view(
    selected_model,
    selected_datasets,
    selected_categories=None,
    min_count=DEFAULT_MIN_COUNT,
):
    empty_fig = EMPTY_FIG()
    empty_df = pd.DataFrame([{"Message": "No data available."}])

    def _blank(message: str):
        return (
            message,
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            empty_df, empty_df, empty_df,
            gr.update(choices=[], value=[]),
            None, empty_df, empty_df,
        )

    if not selected_model or not selected_datasets:
        return _blank("️ Select a model and at least one dataset.")

    raw_df, categories = load_data(selected_model, selected_datasets)
    if raw_df is None or raw_df.empty:
        return _blank(" Cannot load evaluation data.")

    df = _attach_penta_columns(raw_df)

    if selected_categories is not None:
        selected_categories = [c for c in selected_categories if c in categories]
        if not selected_categories and categories:
            selected_categories = categories
    else:
        selected_categories = categories

    view = _filter_df(df, selected_categories)
    min_count = max(1, int(min_count or DEFAULT_MIN_COUNT))
    pair_min = min(min_count, PAIR_MIN_COUNT)

    kpi = generate_kpi_text(view)
    fig_pie = fig_overall_response_pie(view)
    fig_category = fig_category_vulnerability(view)
    fig_density_response = fig_component_counts_by_response(view)
    fig_density_category = fig_total_density_by_category(view)
    fig_density_heatmap = fig_component_density_heatmap(view)
    fig_structural = fig_structural_configuration(view, min_count=pair_min)
    fig_components = fig_top_individual_components(view, min_count=pair_min, top_k=12)
    fig_domain = fig_domain_response_ratio(view, min_count=min_count)
    fig_domain_top = fig_domain_top_component(view, min_count=min_count)
    fig_incremental = fig_incremental_contribution(view, min_count=pair_min)
    module_pairs, component_pairs = get_pairwise_data(view, min_count=pair_min)
    raw_table = prep_table(raw_df, selected_categories)

    return (
        kpi,
        fig_pie,
        fig_category,
        fig_density_response,
        fig_density_category,
        fig_density_heatmap,
        fig_structural,
        fig_components,
        fig_domain,
        fig_domain_top,
        fig_incremental,
        module_pairs,
        component_pairs,
        raw_table,
        gr.update(choices=categories, value=selected_categories),
        view,
        module_pairs,
        component_pairs,
    )


def handle_model_change(model, min_count):
    datasets = get_dataset_choices(model) or []
    result = on_update_view(model, datasets, None, min_count)
    return (gr.update(choices=datasets, value=datasets),) + result


# UI
def render_analysis_tab(shared_config: gr.State):
    init_models = get_model_choices()
    init_model = init_models[0] if init_models else None
    init_datasets = get_dataset_choices(init_model) if init_model else []
    init_datasets = init_datasets or []

    initial = on_update_view(init_model, init_datasets, None, DEFAULT_MIN_COUNT)

    current_df = gr.State(None)
    current_module_pairs = gr.State(None)
    current_component_pairs = gr.State(None)

    gr.Markdown("##  PENTA Structural Vulnerability Analytics")
    gr.Markdown(
        "**PENTA** = **Role · Domain · Action · Object · Format**.\n\n"
        "Baseline → density → configuration → components → domain → pairwise → conditional contribution.\n\n"
        "> Association only — not causal effects. All rates below carry 95% Wilson confidence intervals where noted."
    )

    with gr.Row(equal_height=True):
        model_dropdown = gr.Dropdown(
            label="1. Target Model", choices=init_models, value=init_model, interactive=True, scale=2
        )
        dataset_checkboxes = gr.CheckboxGroup(
            label="2. Datasets", choices=init_datasets, value=init_datasets, interactive=True, scale=4
        )
        min_count_slider = gr.Slider(
            label="Minimum Support (N)",
            minimum=1,
            maximum=30,
            value=DEFAULT_MIN_COUNT,
            step=1,
            interactive=True,
            scale=2,
        )
        refresh_btn = gr.Button(" Sync Data", scale=1)

    cat_init = initial[14] if isinstance(initial[14], dict) else {}
    category_checkboxes = gr.CheckboxGroup(
        label="3. Filter Threat Categories",
        choices=cat_init.get("choices", []),
        value=cat_init.get("value", []),
        interactive=True,
    )
    kpi_markdown = gr.Markdown(value=initial[0])

    gr.Markdown("---\n### 1. Baseline Jailbreak Vulnerability")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=320):
            chart_pie = gr.Plot(value=initial[1])
            gr.Markdown(" *Overall Jailbroken vs Refused.*")
        with gr.Column(scale=1, min_width=320):
            chart_category = gr.Plot(value=initial[2])
            gr.Markdown(" *Jailbreak rate by threat category, with 95% CI error bars.*")

    gr.Markdown("---\n### 2. PENTA Component Density")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=320):
            chart_density_response = gr.Plot(value=initial[3])
            gr.Markdown(" *Mean component count · Response vs Refusal.*")
        with gr.Column(scale=1, min_width=320):
            chart_density_category = gr.Plot(value=initial[4])
            gr.Markdown(" *Mean total components · Category × Outcome.*")
    chart_density_heatmap = gr.Plot(value=initial[5])
    gr.Markdown(" *Response-only component density by threat category.*")

    gr.Markdown("---\n### 3. Structural Configuration")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=320):
            chart_structural = gr.Plot(value=initial[6])
            gr.Markdown(" *Bayesian-smoothed rate by module-presence configuration.*")
        with gr.Column(scale=1, min_width=320):
            chart_components = gr.Plot(value=initial[7])
            gr.Markdown(" *Top components by jailbreak association, vs. global baseline.*")

    gr.Markdown("---\n### 4. Domain-Centric Analysis")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=320):
            chart_domain = gr.Plot(value=initial[8])
            gr.Markdown(" *Domain-wise jailbreak / refusal rate.*")
        with gr.Column(scale=1, min_width=320):
            chart_domain_top = gr.Plot(value=initial[9])
            gr.Markdown(" *Domain × highest-associated non-domain component.*")

    gr.Markdown("---\n### 5. Conditional / Incremental Module Contribution")
    chart_incremental = gr.Plot(value=initial[10])
    gr.Markdown(" *How much does adding a second module shift the jailbreak rate vs. the base module alone? Red = amplifies risk, blue = dampens it.*")

    gr.Markdown("---\n### 6. High-Risk Pairwise Combinations")
    gr.Markdown("Click any row to inspect the underlying successful jailbreak prompts below.")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=320):
            module_pairs_table = gr.Dataframe(
                value=initial[11], label="Module-type pairs", interactive=False
            )
        with gr.Column(scale=1, min_width=320):
            component_pairs_table = gr.Dataframe(
                value=initial[12], label="Cross-module component pairs", interactive=False
            )

    gr.Markdown("###  Prompt-level Inspection")
    prompt_viewer = gr.Dataframe(
        value=None, label="Successful jailbreak prompts", interactive=False, wrap=True
    )

    with gr.Accordion("Raw Evaluation Data", open=False):
        data_preview = gr.Dataframe(value=initial[13], interactive=False)

    update_outputs = [
        kpi_markdown,
        chart_pie,
        chart_category,
        chart_density_response,
        chart_density_category,
        chart_density_heatmap,
        chart_structural,
        chart_components,
        chart_domain,
        chart_domain_top,
        chart_incremental,
        module_pairs_table,
        component_pairs_table,
        data_preview,
        category_checkboxes,
        current_df,
        current_module_pairs,
        current_component_pairs,
    ]

    def run_update(model, datasets, categories, min_count):
        return on_update_view(model, datasets, categories, min_count)

    model_dropdown.change(
        fn=handle_model_change,
        inputs=[model_dropdown, min_count_slider],
        outputs=[dataset_checkboxes] + update_outputs,
    )
    for component in (dataset_checkboxes, category_checkboxes, min_count_slider):
        component.change(
            fn=run_update,
            inputs=[model_dropdown, dataset_checkboxes, category_checkboxes, min_count_slider],
            outputs=update_outputs,
        )
    refresh_btn.click(
        fn=handle_model_change,
        inputs=[model_dropdown, min_count_slider],
        outputs=[dataset_checkboxes] + update_outputs,
    )

    module_pairs_table.select(
        fn=handle_pair_click,
        inputs=[current_module_pairs, current_df],
        outputs=[prompt_viewer],
    )
    component_pairs_table.select(
        fn=handle_pair_click,
        inputs=[current_component_pairs, current_df],
        outputs=[prompt_viewer],
    )