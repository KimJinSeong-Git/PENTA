# web/tab_penta_analysis.py
import matplotlib
matplotlib.use("Agg")
import gradio as gr
import pandas as pd
from core.penta_analytics import (
    MODEL_DISPLAY,
    DATASET_DISPLAY,
    MODULES,
    FINGERPRINT_VIEWS,
    get_table1_corpus_structure,
    get_component_catalog,
    get_decomposed_samples,
    get_table5_baseline_refusal,
    get_table6_category_refusal,
    plot_baseline_comparison,
    get_table3_single_risk_ranking,
    plot_single_risk_chart,
    get_table4_pairwise_synergy,
    plot_pairwise_synergy_chart,
    plot_structural_density_curve,
    get_table7_cosine_similarity,
    get_all_15_pairs_similarity_table,
    plot_fingerprint_combined_figure,
    get_model_pair_comparison,
)


def render_penta_comprehensive_analysis_tab(shared_config: gr.State):
    """Render comprehensive analysis dashboard across four analytical perspectives."""

    models_list = ["All"] + list(MODEL_DISPLAY.values())
    single_models_list = list(MODEL_DISPLAY.values())
    modules_list = ["All"] + MODULES
    datasets_list = list(DATASET_DISPLAY.keys())

    with gr.Tabs():
        # Subtab 1: Corpus and module distribution
        with gr.Tab("Corpus & Module Distribution"):
            gr.Markdown("### Corpus Composition & Functional Module Distribution")
            gr.Markdown(
                "Examines the structural distribution of prompt components and functional module density "
                "across the five functional modules: Role, Domain, Action, Object, and Format."
            )

            with gr.Accordion("Corpus Structural Density & Module Distribution", open=True):
                gr.Markdown(
                    "**What you can learn here**: Compare structural complexity across benchmark corpora and real-world jailbreak prompts. "
                    "Direct benchmarks (HarmBench, JBB-Behaviors) focus on core queries (1-2 components), while in-the-wild evasive prompts "
                    "wrap malicious payloads with extensive contextual framing (averaging 12+ components and activating all 5 modules)."
                )
                table1_df = gr.Dataframe(
                    value=get_table1_corpus_structure(),
                    interactive=False,
                    wrap=True,
                )

            with gr.Accordion("Extracted Component Catalog Browser", open=True):
                gr.Markdown(
                    "**What you can learn here**: Browse the distinct vocabulary of functional components extracted "
                    "per module across different prompt corpora."
                )
                catalog_dataset_select = gr.Dropdown(
                    label="Select Dataset",
                    choices=[("HarmBench Dataset", "HarmBench"),
                             ("JBB-Behaviors Dataset", "JBB-Behaviors"),
                             ("In-the-wild Prompts", "in-the-wild-jailbreak-prompts")],
                    value="HarmBench",
                    interactive=True,
                )
                catalog_output = gr.Markdown()

                def _update_catalog(ds_key):
                    cat = get_component_catalog(ds_key)
                    md_text = f"#### Component Catalog: {DATASET_DISPLAY.get(ds_key, ds_key)}\n\n"
                    for mod in MODULES:
                        items = cat.get(mod, [])
                        md_text += f"**{mod} ({len(items)} unique)**:\n"
                        md_text += ", ".join(items) if items else "(None)"
                        md_text += "\n\n"
                    return md_text

                catalog_dataset_select.change(
                    fn=_update_catalog,
                    inputs=[catalog_dataset_select],
                    outputs=[catalog_output],
                )
                catalog_output.value = _update_catalog("HarmBench")

            with gr.Accordion("Decomposed Prompt Sample Inspector", open=True):
                gr.Markdown(
                    "**What you can learn here**: Inspect raw adversarial prompts mapped verbatim into structured module elements "
                    "(Role, Domain, Action, Object, Format)."
                )
                samples_dataset_select = gr.Dropdown(
                    label="Select Dataset",
                    choices=[("HarmBench Samples", "HarmBench"),
                             ("JBB-Behaviors Samples", "JBB-Behaviors"),
                             ("In-the-wild Samples", "in-the-wild-jailbreak-prompts")],
                    value="HarmBench",
                    interactive=True,
                )
                samples_df = gr.Dataframe(
                    value=pd.DataFrame(get_decomposed_samples("HarmBench", limit=8)),
                    interactive=False,
                    wrap=True,
                )

                def _update_samples(ds_key):
                    return pd.DataFrame(get_decomposed_samples(ds_key, limit=8))

                samples_dataset_select.change(
                    fn=_update_samples,
                    inputs=[samples_dataset_select],
                    outputs=[samples_df],
                )

        # Subtab 2: Baseline model vulnerability
        with gr.Tab("Baseline Model Vulnerability"):
            gr.Markdown("### Baseline Defense Performance & Threat Category Profile")
            gr.Markdown(
                "Evaluates standard prompt-level defense performance across models and datasets under greedy decoding "
                "(Temperature 0.0, Top-p 1.0)."
            )

            with gr.Row():
                metric_toggle = gr.Radio(
                    label="Display Metric",
                    choices=["Refusal Rate (%)", "Attack Success Rate (ASR %)"],
                    value="Refusal Rate (%)",
                    interactive=True,
                )

            with gr.Accordion("Aggregate Defense Performance by Model & Dataset", open=True):
                gr.Markdown(
                    "**What you can learn here**: Benchmark the overall defense strength of each model across direct benchmark attacks "
                    "versus multi-component in-the-wild jailbreaks."
                )
                table5_df = gr.Dataframe(
                    value=get_table5_baseline_refusal(as_asr=False),
                    interactive=False,
                    wrap=True,
                )
                baseline_chart = gr.Plot(value=plot_baseline_comparison(as_asr=False))

            def _toggle_baseline_metric(metric_val):
                as_asr = (metric_val == "Attack Success Rate (ASR %)")
                return (
                    get_table5_baseline_refusal(as_asr=as_asr),
                    plot_baseline_comparison(as_asr=as_asr),
                )

            metric_toggle.change(
                fn=_toggle_baseline_metric,
                inputs=[metric_toggle],
                outputs=[table5_df, baseline_chart],
            )

            with gr.Accordion("Fine-Grained Defense Breakdown by Threat Category", open=True):
                gr.Markdown(
                    "**What you can learn here**: Identify model-specific domain blind spots. Pinpoint which threat categories "
                    "(e.g. Cyberattacks, Chemical Hazards, Misinformation) maintain robust safeguards and which are susceptible to bypass."
                )
                category_ds_select = gr.Radio(
                    label="Dataset Split",
                    choices=[("HarmBench (Standard Threat Domains)", "HarmBench"),
                             ("JBB-Behaviors (Detailed Threat Domains)", "JBB-Behaviors")],
                    value="HarmBench",
                    interactive=True,
                )
                table6_df = gr.Dataframe(
                    value=get_table6_category_refusal("HarmBench"),
                    interactive=False,
                    wrap=True,
                )

                category_ds_select.change(
                    fn=lambda ds: get_table6_category_refusal(ds),
                    inputs=[category_ds_select],
                    outputs=[table6_df],
                )

        # Subtab 3: Structural risk and synergy
        with gr.Tab("Structural Risk & Synergy"):
            gr.Markdown("### Structural Risk Amplification & Multi-Component Synergy")
            gr.Markdown(
                "Isolates which structural elements and component combinations trigger safety failures, "
                "uncovering single-component vulnerabilities and multi-component bypass synergies."
            )

            with gr.Accordion("Single-Component Risk Rankings (Refusal Lift Analysis)", open=True):
                gr.Markdown(
                    "**What you can learn here**: Discover which individual prompt components cause the largest drop in model refusal rates "
                    "(Delta pp relative to model baseline). High negative values reveal critical vulnerability surfaces; positive values indicate defensive over-refusal."
                )
                with gr.Row():
                    t3_model_sel = gr.Dropdown(
                        label="Target Model",
                        choices=models_list,
                        value="Claude Sonnet 4.5",
                        interactive=True,
                        scale=2,
                    )
                    t3_mod_filter = gr.Dropdown(
                        label="Module Filter",
                        choices=modules_list,
                        value="All",
                        interactive=True,
                        scale=2,
                    )
                    t3_support = gr.Slider(
                        label="Min Support (n)",
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                        interactive=True,
                        scale=2,
                    )
                    t3_topk = gr.Slider(
                        label="Top K",
                        minimum=3,
                        maximum=15,
                        value=5,
                        step=1,
                        interactive=True,
                        scale=1,
                    )

                table3_df = gr.Dataframe(
                    value=get_table3_single_risk_ranking(
                        model_name="Claude Sonnet 4.5", module_filter="All", min_support=10, top_k=5
                    ),
                    interactive=False,
                    wrap=True,
                )
                t3_chart = gr.Plot(
                    value=plot_single_risk_chart(
                        model_name="Claude Sonnet 4.5", module_filter="All", min_support=10, top_k=5
                    )
                )

                def _update_t3(m_name, m_filt, sup, topk):
                    return (
                        get_table3_single_risk_ranking(model_name=m_name, module_filter=m_filt, min_support=int(sup), top_k=int(topk)),
                        plot_single_risk_chart(model_name=(m_name if m_name != "All" else "Claude Sonnet 4.5"), module_filter=m_filt, min_support=int(sup), top_k=int(topk)),
                    )

                for comp in [t3_model_sel, t3_mod_filter, t3_support, t3_topk]:
                    comp.change(fn=_update_t3, inputs=[t3_model_sel, t3_mod_filter, t3_support, t3_topk], outputs=[table3_df, t3_chart])

            with gr.Accordion("Pairwise Component Synergy Auditor", open=True):
                gr.Markdown(
                    "**What you can learn here**: Detect synergistic combinations where pairing two components bypasses safeguards "
                    "far more effectively than either component alone. Evaluated using Fisher's exact test with Benjamini-Hochberg FDR control."
                )
                with gr.Row():
                    t4_model_sel = gr.Dropdown(
                        label="Target Model",
                        choices=models_list,
                        value="Claude Sonnet 4.5",
                        interactive=True,
                        scale=2,
                    )
                    t4_topk = gr.Slider(
                        label="Top Pairs",
                        minimum=3,
                        maximum=10,
                        value=3,
                        step=1,
                        interactive=True,
                        scale=1,
                    )

                table4_df = gr.Dataframe(
                    value=get_table4_pairwise_synergy(model_name="Claude Sonnet 4.5", top_k=3),
                    interactive=False,
                    wrap=True,
                )
                t4_chart = gr.Plot(value=plot_pairwise_synergy_chart(model_name="Claude Sonnet 4.5", top_k=3))

                def _update_t4(m_name, topk):
                    target_m = m_name if m_name != "All" else "Claude Sonnet 4.5"
                    return (
                        get_table4_pairwise_synergy(model_name=m_name, top_k=int(topk)),
                        plot_pairwise_synergy_chart(model_name=target_m, top_k=int(topk)),
                    )

                for comp in [t4_model_sel, t4_topk]:
                    comp.change(fn=_update_t4, inputs=[t4_model_sel, t4_topk], outputs=[table4_df, t4_chart])

            with gr.Accordion("Structural Complexity vs Safety Degradation Profile", open=True):
                gr.Markdown(
                    "**What you can learn here**: Observe how prompt structural density (total component count) systematically "
                    "degrades LLM refusal rates across different model architectures."
                )
                density_plot = gr.Plot(value=plot_structural_density_curve())

        # Subtab 4: Alignment sensitivity fingerprints
        with gr.Tab("Alignment Sensitivity Fingerprints"):
            gr.Markdown("### Cross-Model Alignment Sensitivity Fingerprints & Comparative Audit")
            gr.Markdown(
                "Profiles whether evaluated LLMs share common structural vulnerabilities and groups models "
                "by alignment sensitivity patterns."
            )

            with gr.Row():
                view_selector = gr.Dropdown(
                    label="Fingerprint View",
                    choices=FINGERPRINT_VIEWS,
                    value="Pairwise: Integrated (Full Synergy)",
                    interactive=True,
                    scale=3,
                )

            with gr.Accordion("Sensitivity Correlation Matrix & Hierarchical Clustering", open=True):
                gr.Markdown(
                    "**What you can learn here**: Analyze pairwise alignment similarity and hierarchical groupings (Ward linkage) "
                    "across proprietary (closed-weight) versus public (open-weight) models."
                )
                fig2_plot = gr.Plot(value=plot_fingerprint_combined_figure("Pairwise: Integrated (Full Synergy)"))

            view_selector.change(
                fn=plot_fingerprint_combined_figure,
                inputs=[view_selector],
                outputs=[fig2_plot],
            )

            with gr.Accordion("Comprehensive Pairwise Sensitivity Index (All 15 Model Pairs)", open=True):
                gr.Markdown(
                    "**What you can learn here**: Inspect correlation coefficients across all 15 model pairs for 6 distinct sensitivity views, "
                    "contrasting shared vulnerability patterns with idiosyncratic over-refusal behaviors."
                )
                table7_df = gr.Dataframe(
                    value=get_all_15_pairs_similarity_table(),
                    interactive=False,
                    wrap=True,
                )

            with gr.Accordion("Head-to-Head Model Vulnerability Comparison", open=True):
                gr.Markdown(
                    "**What you can learn here**: Compare two models side-by-side on specific structural features to identify "
                    "shared weaknesses and model-unique blind spots."
                )
                with gr.Row():
                    aud_m1 = gr.Dropdown(
                        label="Model A",
                        choices=single_models_list,
                        value="Gemini 3.1 Pro",
                        interactive=True,
                    )
                    aud_m2 = gr.Dropdown(
                        label="Model B",
                        choices=single_models_list,
                        value="Llama-3.1-8B",
                        interactive=True,
                    )
                    aud_view = gr.Dropdown(
                        label="Auditing View",
                        choices=FINGERPRINT_VIEWS,
                        value="Single: Integrated (Full Structure)",
                        interactive=True,
                    )

                aud_summary = gr.Markdown()
                aud_chart = gr.Plot()
                aud_table = gr.Dataframe(interactive=False, wrap=True)

                def _audit_models(m1, m2, v):
                    sim_val, df_diff, fig = get_model_pair_comparison(m1, m2, v)
                    md_text = f"**Pairwise Cosine Similarity**: `{sim_val:.3f}` | **Comparing**: `{m1}` vs `{m2}` ({v})"
                    return md_text, fig, df_diff

                init_md, init_fig, init_tbl = _audit_models("Gemini 3.1 Pro", "Llama-3.1-8B", "Single: Integrated (Full Structure)")
                aud_summary.value = init_md
                aud_chart.value = init_fig
                aud_table.value = init_tbl

                for comp in [aud_m1, aud_m2, aud_view]:
                    comp.change(
                        fn=_audit_models,
                        inputs=[aud_m1, aud_m2, aud_view],
                        outputs=[aud_summary, aud_chart, aud_table],
                    )


render_penta_analysis_tab = render_penta_comprehensive_analysis_tab
