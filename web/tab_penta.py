# web/tab_penta.py
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data_utils import get_model_choices
from web.tab_basic_analysis import (
    _all_datasets,
    _empty,
    _pct,
    _prompt_count,
    attach_penta_columns,
    load_data,
)

try:
    from web.tab_basic_analysis import _blank_fig
except ImportError:
    def _blank_fig(msg: str = "No data available."):
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        fig.tight_layout()
        return fig


SLICE_ORDER = [
    "No Role, no Format",
    "Role only",
    "Format only",
    "Role + Format",
]
MODULE_COLS = ["Role", "Domain", "Action", "Object", "Format"]
TOP_K = 6
TOP_DELTA = 5
TOTAL_CAP = 12


def _role_format_slice(mods) -> str:
    mods = mods or []
    has_role = "Role" in mods
    has_format = "Format" in mods
    if has_role and has_format:
        return "Role + Format"
    if has_role:
        return "Role only"
    if has_format:
        return "Format only"
    return "No Role, no Format"


def _as_list(val):
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    text = str(val).strip()
    return [text] if text else []


def attach_slices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RF_Slice"] = out["Present_Modules"].map(_role_format_slice)
    for col in MODULE_COLS:
        out[f"n_{col}"] = out[col].map(_as_list).map(len) if col in out.columns else 0
    out["Total_Components"] = out[[f"n_{c}" for c in MODULE_COLS]].sum(axis=1)
    return out


def _slice_counts(df: pd.DataFrame) -> pd.Series:
    if "Prompt" in df.columns:
        return df.groupby("RF_Slice")["Prompt"].nunique()
    return df.groupby("RF_Slice").size()


def _models(df: pd.DataFrame) -> list:
    return list(df["Model"].dropna().unique()) if df is not None and not df.empty else []


def summarize_rf_slices(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()
    models = _models(df)
    rows = []
    for name in SLICE_ORDER:
        sdf = df.loc[df["RF_Slice"] == name]
        row = {"Slice": name, "n": _prompt_count(sdf)}
        for model in models:
            mdf = sdf.loc[sdf["Model"] == model]
            row[model] = _pct(mdf["Is_Jailbroken"].mean()) if len(mdf) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def plot_slice_counts(df: pd.DataFrame):
    if df is None or df.empty:
        return _blank_fig()
    g = _slice_counts(df).reindex(SLICE_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.bar(g.index, g.values, color="#4C78A8")
    ax.set_ylabel("n prompts")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(g.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def plot_slice_asr(df: pd.DataFrame):
    if df is None or df.empty:
        return _blank_fig()
    models = _models(df)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    x = np.arange(len(SLICE_ORDER))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        sdf = df.loc[df["Model"] == model]
        rates = []
        for name in SLICE_ORDER:
            sl = sdf.loc[sdf["RF_Slice"] == name]
            rates.append(float(sl["Is_Jailbroken"].mean() * 100) if len(sl) else 0.0)
        ax.bar(x + i * width, rates, width=width, label=model)
    ax.set_xticks(x + width * max(len(models) - 1, 0) / 2)
    ax.set_xticklabels(SLICE_ORDER, rotation=15)
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_model_slice_heatmap(df: pd.DataFrame):
    if df is None or df.empty:
        return _blank_fig()
    asr = (
        df.groupby(["RF_Slice", "Model"])["Is_Jailbroken"]
        .mean()
        .mul(100)
        .unstack("Model")
        .reindex(SLICE_ORDER)
    )
    if "Prompt" in df.columns:
        ntab = df.groupby(["RF_Slice", "Model"])["Prompt"].nunique()
    else:
        ntab = df.groupby(["RF_Slice", "Model"]).size()
    ntab = ntab.unstack("Model").reindex(SLICE_ORDER)
    asr = asr.fillna(0.0)
    ntab = ntab.fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    im = ax.imshow(asr.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(asr.columns)))
    ax.set_xticklabels(list(asr.columns), rotation=18, ha="right")
    ax.set_yticks(range(len(asr.index)))
    ax.set_yticklabels(list(asr.index))
    ax.grid(False)
    for i in range(asr.shape[0]):
        for j in range(asr.shape[1]):
            val = asr.values[i, j]
            cnt = ntab.values[i, j] if i < ntab.shape[0] and j < ntab.shape[1] else 0
            color = "white" if val >= 55 else "#1b1b1b"
            ax.text(j, i, f"{val:.1f}%\nn={int(cnt)}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="ASR (%)")
    fig.tight_layout()
    return fig


def _explode_labels(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()
    keep = [c for c in ["Model", "Dataset", "Prompt", "Is_Jailbroken", col] if c in df.columns]
    x = df[keep].copy()
    x[col] = x[col].map(_as_list)
    x = x.explode(col)
    x = x.loc[x[col].notna() & (x[col].astype(str).str.len() > 0)]
    return x


def _sort_rows(frame: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "n":
        return frame.sort_values(["_n", "_asr"], ascending=[False, False])
    return frame.sort_values(["_asr", "_n"], ascending=[False, False])


def summarize_label_asr(
    df: pd.DataFrame, col: str, min_n: int = 1, top_k: int = TOP_K, sort_by: str = "n"
) -> pd.DataFrame:
    x = _explode_labels(df, col)
    if x.empty:
        return _empty(f"No {col} labels.")
    rows = []
    for label, g in x.groupby(col, dropna=False):
        n = _prompt_count(g)
        if n < min_n:
            continue
        rate = float(g["Is_Jailbroken"].mean())
        rows.append({col: label, "n": n, "ASR": _pct(rate), "_asr": rate, "_n": n})
    if not rows:
        return _empty(f"No {col} label with n ≥ {min_n}.")
    return (
        _sort_rows(pd.DataFrame(rows), sort_by)
        .head(top_k)
        .drop(columns=["_asr", "_n"])
        .reset_index(drop=True)
    )


def summarize_role_format_pairs(
    df: pd.DataFrame, min_n: int = 1, top_k: int = TOP_K, sort_by: str = "n"
) -> pd.DataFrame:
    if df is None or df.empty or "Role" not in df.columns or "Format" not in df.columns:
        return _empty("No Role×Format pairs.")
    recs = []
    for _, row in df.iterrows():
        roles = _as_list(row.get("Role"))
        formats = _as_list(row.get("Format"))
        if not roles or not formats:
            continue
        for role in roles:
            for fmt in formats:
                recs.append({
                    "Role": role,
                    "Format": fmt,
                    "Model": row.get("Model"),
                    "Prompt": row.get("Prompt"),
                    "Is_Jailbroken": row.get("Is_Jailbroken"),
                })
    if not recs:
        return _empty("No Role×Format pairs.")
    x = pd.DataFrame(recs)
    rows = []
    for (role, fmt), g in x.groupby(["Role", "Format"], dropna=False):
        n = _prompt_count(g)
        if n < min_n:
            continue
        rate = float(g["Is_Jailbroken"].mean())
        rows.append({
            "Role": role,
            "Format": fmt,
            "n": n,
            "ASR": _pct(rate),
            "_asr": rate,
            "_n": n,
        })
    if not rows:
        return _empty(f"No Role×Format pair with n ≥ {min_n}.")
    return (
        _sort_rows(pd.DataFrame(rows), sort_by)
        .head(top_k)
        .drop(columns=["_asr", "_n"])
        .reset_index(drop=True)
    )


def _baselines(df: pd.DataFrame) -> dict:
    return df.groupby("Model")["Is_Jailbroken"].mean().to_dict()


def _fmt_delta(asr, base):
    if asr != asr:
        return ""
    return f"{asr * 100:.1f}% ({(asr - base) * 100:+.1f})"


def _delta_cols(exploded: pd.DataFrame, key: str, baselines: dict, min_n: int, k: int):
    items = []
    models = list(baselines)
    for label, g in exploded.groupby(key, dropna=False):
        if _prompt_count(g) < min_n:
            continue
        deltas = []
        asrs = {}
        for model in models:
            sl = g.loc[g["Model"] == model]
            if sl.empty:
                continue
            asr = float(sl["Is_Jailbroken"].mean())
            asrs[model] = asr
            deltas.append(asr - baselines[model])
        if len(deltas) < 2:
            continue
        items.append((str(label), float(np.std(deltas)), asrs))
    items.sort(key=lambda t: t[1], reverse=True)
    return items[:k]


def summarize_delta_matrix(df: pd.DataFrame, col: str, min_n: int) -> pd.DataFrame:
    x = _explode_labels(df, col)
    if x.empty:
        return _empty(f"No {col} labels.")
    bases = _baselines(df)
    picked = _delta_cols(x, col, bases, min_n, TOP_DELTA)
    if not picked:
        return _empty(f"No {col} label with n ≥ {min_n} and ≥2 models.")
    rows = []
    for model, base in bases.items():
        row = {"Model": model}
        for lab, _std, asrs in picked:
            row[lab] = _fmt_delta(asrs.get(model, float("nan")), base)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_pair_delta_matrix(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty("No Role×Format pairs.")
    recs = []
    for _, row in df.iterrows():
        roles = _as_list(row.get("Role"))
        formats = _as_list(row.get("Format"))
        if not roles or not formats:
            continue
        for role in roles:
            for fmt in formats:
                recs.append({
                    "pair": f"{role} × {fmt}",
                    "Model": row.get("Model"),
                    "Prompt": row.get("Prompt"),
                    "Is_Jailbroken": row.get("Is_Jailbroken"),
                })
    if not recs:
        return _empty("No Role×Format pairs.")
    x = pd.DataFrame(recs)
    bases = _baselines(df)
    picked = _delta_cols(x, "pair", bases, min_n, TOP_DELTA)
    if not picked:
        return _empty("No pair with enough n.")
    rows = []
    for model, base in bases.items():
        row = {"Model": model}
        for lab, _std, asrs in picked:
            row[lab] = _fmt_delta(asrs.get(model, float("nan")), base)
        rows.append(row)
    return pd.DataFrame(rows)


def _bin_total(series: pd.Series) -> pd.Series:
    cap = series.clip(upper=TOTAL_CAP)
    return cap.map(lambda v: f"{TOTAL_CAP}+" if int(v) >= TOTAL_CAP else str(int(v)))


def summarize_total_count(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()
    x = df.copy()
    x["bin"] = _bin_total(x["Total_Components"])
    models = _models(df)
    order = [str(i) for i in range(0, TOTAL_CAP)] + [f"{TOTAL_CAP}+"]
    rows = []
    for name in order:
        sdf = x.loc[x["bin"] == name]
        if sdf.empty:
            continue
        row = {"#components": name}
        for model in models:
            mdf = sdf.loc[sdf["Model"] == model]
            if mdf.empty:
                row[model] = ""
            else:
                rate = float(mdf["Is_Jailbroken"].mean())
                row[model] = f"{rate * 100:.2f}% ({_prompt_count(mdf)})"
        rows.append(row)
    return pd.DataFrame(rows) if rows else _empty()


def plot_total_count_line(df: pd.DataFrame):
    if df is None or df.empty:
        return _blank_fig()
    models = _models(df)
    x = df.copy()
    x["k"] = x["Total_Components"].clip(upper=TOTAL_CAP).astype(int)
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for model in models:
        g = x.loc[x["Model"] == model].groupby("k")["Is_Jailbroken"].mean().mul(100)
        ax.plot(g.index, g.values, marker="o", label=model)
    ax.set_xlabel("# components in prompt")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if len(models) > 1:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def _presence_views(df: pd.DataFrame):
    x = attach_slices(df)
    return (
        summarize_rf_slices(x),
        plot_slice_counts(x),
        plot_slice_asr(x),
        plot_model_slice_heatmap(x),
    )


def _label_views_model(df: pd.DataFrame, min_n: int, sort_by: str):
    x = attach_slices(df)
    empty = _empty("No rows.")
    if x is None or x.empty:
        return empty, empty, empty, empty, empty, empty
    return (
        summarize_label_asr(x, "Domain", min_n, sort_by=sort_by),
        summarize_label_asr(x, "Action", min_n, sort_by=sort_by),
        summarize_label_asr(x, "Object", min_n, sort_by=sort_by),
        summarize_label_asr(x, "Role", min_n, sort_by=sort_by),
        summarize_label_asr(x, "Format", min_n, sort_by=sort_by),
        summarize_role_format_pairs(x, min_n, sort_by=sort_by),
    )


def _label_views_total(df: pd.DataFrame, min_n: int):
    x = attach_slices(df)
    empty = _empty("No rows.")
    if x is None or x.empty:
        return empty, empty, empty, empty, empty, empty
    return (
        summarize_delta_matrix(x, "Domain", min_n),
        summarize_delta_matrix(x, "Action", min_n),
        summarize_delta_matrix(x, "Object", min_n),
        summarize_delta_matrix(x, "Role", min_n),
        summarize_delta_matrix(x, "Format", min_n),
        summarize_pair_delta_matrix(x, min_n),
    )


def _count_views(df: pd.DataFrame):
    x = attach_slices(df)
    empty = _empty("No rows.")
    fig = _blank_fig("No rows.")
    if x is None or x.empty:
        return empty, fig
    return summarize_total_count(x), plot_total_count_line(x)


def _empty_presence():
    empty = _empty()
    fig = _blank_fig("Not selected.")
    return [empty, fig, fig, fig]


def _empty_detail():
    empty = _empty()
    fig = _blank_fig("Not selected.")
    return [empty, empty, empty, empty, empty, empty, empty, fig]


def _empty_page(n_slots: int):
    pack = _empty_presence()
    for _ in range(n_slots):
        pack.extend(_empty_detail())
    return pack


def _label_block():
    return (
        gr.Dataframe(label="Domain", interactive=False, wrap=True),
        gr.Dataframe(label="Action", interactive=False, wrap=True),
        gr.Dataframe(label="Object / Goal", interactive=False, wrap=True),
        gr.Dataframe(label="Role", interactive=False, wrap=True),
        gr.Dataframe(label="Format", interactive=False, wrap=True),
        gr.Dataframe(label="Role × Format", interactive=False, wrap=True),
    )


def _count_block():
    return (
        gr.Dataframe(label="ASR by total #components", interactive=False, wrap=True),
        gr.Plot(label="ASR vs total #components"),
    )


def _model_subtabs(models):
    comps = []
    with gr.Tabs():
        with gr.Tab("Total"):
            gr.Markdown(
                "#### 2. Labels with the most model-to-model Δ "
                "(cell = ASR (Δ vs that model's overall ASR))"
            )
            comps.extend(_label_block())
            gr.Markdown("#### 3. Component count")
            comps.extend(_count_block())
        for model in models:
            with gr.Tab(str(model)):
                gr.Markdown(f"#### 2. Label analysis — {model}")
                comps.extend(_label_block())
                gr.Markdown(f"#### 3. Component count — {model}")
                comps.extend(_count_block())
    return comps


def render_penta_tab(shared_config: gr.State):
    init_models = get_model_choices() or []
    init_datasets = _all_datasets(init_models)
    n_slots = len(init_models) + 1

    gr.Markdown("## PENTA configuration analysis")

    with gr.Row():
        model_checkboxes = gr.CheckboxGroup(
            label="Target Models",
            choices=init_models,
            value=init_models,
            interactive=True,
            scale=3,
        )
        dataset_checkboxes = gr.CheckboxGroup(
            label="Datasets",
            choices=init_datasets,
            value=init_datasets,
            interactive=True,
            scale=3,
        )
        load_btn = gr.Button("Load", scale=1)

    dataset_tabs = []
    with gr.Tabs():
        with gr.Tab("Overview"):
            gr.Markdown("### 1. Presence slices")
            slice_asr = gr.Dataframe(interactive=False, wrap=True)
            with gr.Row():
                slice_n_plot = gr.Plot(label="n by Role/Format slice")
                slice_asr_plot = gr.Plot(label="ASR by model × slice")
            heat_plot = gr.Plot(label="Model × Role/Format slice")
            with gr.Row():
                sort_radio = gr.Radio(
                    label="Sort labels by (model tabs)",
                    choices=["n", "ASR"],
                    value="n",
                    interactive=True,
                )
                min_n = gr.Number(label="min n", value=20, precision=0)
            overview_comps = [slice_asr, slice_n_plot, slice_asr_plot, heat_plot]
            overview_comps.extend(_model_subtabs(init_models))

        for dataset_name in init_datasets:
            with gr.Tab(dataset_name):
                gr.Markdown(f"### 1. Presence slices — {dataset_name}")
                t1 = gr.Dataframe(interactive=False, wrap=True)
                with gr.Row():
                    p1 = gr.Plot()
                    p2 = gr.Plot()
                p4 = gr.Plot()
                page = [t1, p1, p2, p4]
                page.extend(_model_subtabs(init_models))
                dataset_tabs.append((dataset_name, page))

    all_outputs = overview_comps + [c for tab in dataset_tabs for c in tab[1]]

    def on_models_change(models):
        names = _all_datasets(models)
        return gr.update(choices=names, value=names)

    def _page_views(df, selected_models, min_n_int, sort_key):
        out = list(_presence_views(df))
        out.extend(list(_label_views_total(df, min_n_int)) + list(_count_views(df)))
        for model in init_models:
            if model not in selected_models:
                out.extend(_empty_detail())
                continue
            sdf = df.loc[df["Model"] == model]
            out.extend(list(_label_views_model(sdf, min_n_int, sort_key)) + list(_count_views(sdf)))
        return out

    def update_tables(models, datasets, sort_by, min_n_val):
        try:
            min_n_int = int(min_n_val or 20)
        except Exception:
            min_n_int = 20
        sort_key = "n" if sort_by != "ASR" else "ASR"
        selected_models = set(models or [])
        blank = []
        for _ in dataset_tabs:
            blank.extend(_empty_page(n_slots))
        df, _ = load_data(models, datasets)
        if df is None or df.empty:
            return tuple(_empty_page(n_slots) + blank)
        parsed = attach_penta_columns(df)
        result = _page_views(parsed, selected_models, min_n_int, sort_key)
        selected_ds = set(datasets or [])
        for name, *_rest in dataset_tabs:
            if name not in selected_ds:
                result.extend(_empty_page(n_slots))
                continue
            result.extend(
                _page_views(
                    parsed.loc[parsed["Dataset"] == name],
                    selected_models,
                    min_n_int,
                    sort_key,
                )
            )
        return tuple(result)

    model_checkboxes.change(
        fn=on_models_change,
        inputs=[model_checkboxes],
        outputs=[dataset_checkboxes],
    )
    load_btn.click(
        fn=update_tables,
        inputs=[model_checkboxes, dataset_checkboxes, sort_radio, min_n],
        outputs=all_outputs,
    )
    sort_radio.change(
        fn=update_tables,
        inputs=[model_checkboxes, dataset_checkboxes, sort_radio, min_n],
        outputs=all_outputs,
    )


render_penta_analysis_tab = render_penta_tab