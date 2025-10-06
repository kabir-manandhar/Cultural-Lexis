# model_analysis/post/cli.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer

from .affect_conc import run_affect_concreteness
from .hypothesis import run_hypothesis_tests, WilcoxonConfig
from .tension import TensionConfig, run_tension_analysis

app = typer.Typer(name="post", help="Post-analysis: affect/concreteness, hypothesis tests, tension set")


# --------------------------- helpers ---------------------------

def _parse_tag_path(items: List[str]) -> Dict[str, Path]:
    """
    Parse repeated --model TAG=PATH items into {tag: Path}.
    """
    out: Dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"Expected TAG=PATH, got: {item}")
        tag, p = item.split("=", 1)
        tag = tag.strip()
        path = Path(p.strip())
        if not tag:
            raise typer.BadParameter(f"Empty tag in: {item}")
        out[tag] = path
    return out


def _parse_pairs(items: List[str]) -> List[Tuple[str, str]]:
    """
    Parse repeated --plot-pair left,right into list of (left, right).
    """
    pairs: List[Tuple[str, str]] = []
    for item in items or []:
        if "," not in item:
            raise typer.BadParameter(f"Expected left,right, got: {item}")
        a, b = [x.strip() for x in item.split(",", 1)]
        if not a or not b:
            raise typer.BadParameter(f"Bad pair: {item}")
        pairs.append((a, b))
    return pairs


# ---------------------- affect / concreteness -------------------

@app.command("affect-conc")
def affect_conc(
    language: str = typer.Option(..., "--language", "-l", help="Language code: en or zh"),
    swow_pickle: Optional[Path] = typer.Option(None, "--swow-pickle", help="Processed SWOW pickle"),
    vad_file: Optional[Path] = typer.Option(None, "--vad", help="VAD lexicon path"),
    concreteness_file: Optional[Path] = typer.Option(None, "--conc", help="Concreteness lexicon path"),
    threshold: float = typer.Option(3.0, "--threshold", help="Concreteness threshold (1–5)"),
    write_json: Optional[Path] = typer.Option(None, "--per-cue-json", help="Write per-cue medians JSON (legacy shape)"),
    out_enriched: Optional[Path] = typer.Option(None, "--out-enriched", help="Write per-word enriched JSON"),
    out_medians: Optional[Path] = typer.Option(None, "--out-medians", help="Write medians-only JSON"),
):
    """
    Compute valence/arousal/(dominance)/concreteness medians, coverage, and optional enriched exports.
    """
    _ = run_affect_concreteness(
        language=language,
        swow_pickle=swow_pickle,
        vad_file=vad_file,
        concreteness_file=concreteness_file,
        threshold=threshold,
        write_json=write_json,
        out_enriched=out_enriched,
        out_medians=out_medians,
    )


# -------------------------- hypothesis -------------------------

@app.command("hypothesis")
def hypothesis(
    language: str = typer.Option(..., "--language", "-l", help="Language code: en or zh"),
    per_cue_json: Path = typer.Option(..., "--per-cue-json", help="Per-cue medians JSON (from affect-conc)"),
    out_csv: Path = typer.Option(..., "--out-csv", help="Where to save Wilcoxon results CSV"),
    min_pairs: int = typer.Option(10, "--min-pairs", help="Minimum paired cues required"),
    include_direct: bool = typer.Option(True, "--include-direct/--no-include-direct", help="Model vs human"),
    include_closeness: bool = typer.Option(True, "--include-closeness/--no-include-closeness", help="Compare closeness to human"),
):
    """
    Paired Wilcoxon tests on cue-level medians. Supports both EN/ZH metric sets.
    """
    cfg = WilcoxonConfig(language=language, min_pairs=min_pairs,
                         include_direct=include_direct, include_closeness=include_closeness)
    _ = run_hypothesis_tests(
        per_cue_json=per_cue_json,
        out_csv=out_csv,
        language=language,
        config=cfg,
    )


# ----------------------------- tension -------------------------

@app.command("tension")
def tension(
    language: str = typer.Option(..., "--language", "-l", help="Language code: en or zh"),
    survey_json: Path = typer.Option(..., "--survey-json", help="WV survey JSON with us_score/china_score"),
    model: List[str] = typer.Option(None, "--model", "-m",
                                    help="Repeat as TAG=PATH for model outputs (JSON with choice_values)"),
    out_dir: Path = typer.Option(Path("out/cross_country"), "--out-dir", help="Output directory"),
    k_top: int = typer.Option(50, "--k-top", help="How many high-divergence questions to keep"),
    plot_pair: List[str] = typer.Option(None, "--plot-pair",
                                        help="Repeat as LEFT_TAG,RIGHT_TAG (must match provided --model tags)"),
    image_format: str = typer.Option("png", "--image-format", help="png or pdf"),
):
    """
    Build the cross-country tension set (top-K by hybrid distance), compute per-model distances,
    write per-model JSONs, vectors CSV, and produce paired composite plots.
    """
    model_outputs = _parse_tag_path(model)
    pairs = _parse_pairs(plot_pair)

    cfg = TensionConfig(
        language=language,
        survey_json=survey_json,
        model_outputs=model_outputs,
        out_dir=out_dir,
        k_top=k_top,
        plot_pairs=pairs if pairs else None,
        image_format=image_format,
    )
    _ = run_tension_analysis(cfg)
