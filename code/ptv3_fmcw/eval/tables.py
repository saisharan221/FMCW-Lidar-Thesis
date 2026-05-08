"""LaTeX table writers (spec §12).

Outputs booktabs-style table fragments meant to be `\\input{}`-ed from
`Report.tex`. Numbers are rendered to 2 decimal places by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ptv3_fmcw.data.class_names import TOP_CLASSES


def _fmt(x: float, fmt: str = ".2f") -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "--"
    return f"{x:{fmt}}"


def generate_T1(
    results: Mapping[str, dict],
    out_path: Path,
    methods: Sequence[str] = ("B0_zero", "B1_doppler_only", "B2_class_mean", "B3_doppler_plus_class_mean"),
    pretty: Mapping[str, str] | None = None,
) -> Path:
    """T1: main results table — overall metrics per method.

    Columns (spec §12.1):
        Method, EPE_all, EPE_dyn, EPE_t, EPE_r, dtheta_dyn, |d|v||_dyn
    """
    pretty = pretty or {
        "B0_zero": "B0 Zero",
        "B1_doppler_only": "B1 Doppler-only",
        "B2_class_mean": "B2 Class-mean",
        "B3_doppler_plus_class_mean": "B3 Dop. + class-tan",
    }
    rows = []
    rows.append(r"\begin{tabular}{lcccccc}")
    rows.append(r"\toprule")
    rows.append(
        r"Method & EPE\textsubscript{all} & EPE\textsubscript{dyn} "
        r"& \textbf{EPE\textsubscript{t}} & EPE\textsubscript{r} "
        r"& $\Delta\theta$\textsubscript{dyn} "
        r"& $|\Delta\|v\||$\textsubscript{dyn} \\"
    )
    rows.append(r"\midrule")
    for name in methods:
        m = results.get(name, {}).get("overall", {})
        label = pretty.get(name, name).replace("_", r"\_")
        rows.append(
            f"{label} & "
            f"{_fmt(m.get('epe_all'))} & "
            f"{_fmt(m.get('epe_dyn'))} & "
            f"{_fmt(m.get('epe_t'))} & "
            f"{_fmt(m.get('epe_r'))} & "
            f"{_fmt(m.get('ang_dyn_deg'), '.1f')} & "
            f"{_fmt(m.get('mag_err_dyn'))} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    return out_path


def generate_T2(
    results: Mapping[str, dict],
    out_path: Path,
    classes: Sequence[str] = TOP_CLASSES,
    methods: Sequence[str] = ("B1_doppler_only", "B3_doppler_plus_class_mean"),
    pretty: Mapping[str, str] | None = None,
) -> Path:
    """T2: per-class EPE_t for B1 and B3 (PTv3 column added later)."""
    pretty = pretty or {
        "B1_doppler_only": "B1",
        "B3_doppler_plus_class_mean": "B3",
    }
    rows = []
    col_spec = "l" + "c" * len(methods)
    rows.append(rf"\begin{{tabular}}{{{col_spec}}}")
    rows.append(r"\toprule")
    header = "Class"
    for m in methods:
        header += " & " + pretty.get(m, m).replace("_", r"\_")
    rows.append(header + r" \\")
    rows.append(r"\midrule")
    for cls in classes:
        line = cls.replace("_", r"\_")
        for m in methods:
            v = results.get(m, {}).get("per_class", {}).get(cls, {}).get("epe_t")
            line += " & " + _fmt(v)
        rows.append(line + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    return out_path
