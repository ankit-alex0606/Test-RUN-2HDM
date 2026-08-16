#!/usr/bin/env python3
"""
Integrated 8-core constraint pipeline for the inverted Type-I 2HDM.

Input:
  Large CSV of points already passing theory constraints.

For each point:
  1) Run 2HDMC CalcPhys (Yukawa Type-I).
  2) Parse S,T,U and apply correlated chi^2 cut.
  3) Only for STU survivors:
       - parse the 2HDMC DECAY blocks
       - construct Type-I HiggsTools predictions
       - run HiggsBounds
       - run HiggsSignals
  4) Delete the temporary SLHA file.

Designed for very large CSVs:
  * streams the input (does not load 2M rows into RAM)
  * multiprocessing
  * chunk-level checkpoint/resume
  * temporary SLHA files are deleted
  * Python standard library for CSV/STU algebra (no numpy/pandas)

HiggsSignals:
  By default HS chi2 is CALCULATED AND SAVED but no arbitrary HS chi2
  rejection threshold is imposed. Set --hs-chi2-max VALUE if you have
  chosen the statistically appropriate threshold for your analysis.

Example 100-point validation:
  python3 run_2HDMC_STU_HiggsTools_stream.py \
    --max-points 100 \
    --ncpu 8 \
    --chunk-size 40 \
    --output-dir constraints_test_100

Full scan:
  python3 run_2HDMC_STU_HiggsTools_stream.py \
    --ncpu 8 \
    --chunk-size 400 \
    --output-dir constraints_full_2M
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ============================================================================
# Defaults for this cluster/workflow
# ============================================================================

DEF_INPUT = (
    "/scratch/user/ankitankit/NewFullParamscan/theory_2M/"
    "Allowed_2HDM_theory_unitarity_2000000.csv"
)

DEF_CALCPHYS = "/scratch/user/ankitankit/2HDMC-1.8.0/CalcPhys"
DEF_HB_DATA = "/scratch/user/ankitankit/hbdataset"
DEF_HS_DATA = "/scratch/user/ankitankit/hsdataset"

# 2HDMC output parser for S/T/U
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
STU_PAT = {
    k: re.compile(r"^\s*%s\s+(%s)\s*$" % (k, FLOAT_RE), re.M)
    for k in "STU"
}

ALIASES = {
    "mh": ("Mh", "mh", "m_h"),
    "mH": ("MH", "mH", "m_H"),
    "mA": ("MA", "mA", "m_A"),
    "mHp": ("MHc", "MHC", "MHplus", "mHp", "mHc"),
    "sinba": (
        "sin_ba", "sinbma", "sin_beta_minus_alpha",
        "sin(beta-alpha)", "sba"
    ),
    "m12sq": ("m12sq", "m12_2", "m12^2", "m_12^2", "mu212", "mu2_12"),
    "tanb": ("tanb", "tan_beta", "tanbeta", "tb"),
    "cba": ("cosbma", "cos_ba", "cos(beta-alpha)", "cba"),
}


# ============================================================================
# Small utilities
# ============================================================================

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def finite_float(x):
    y = float(str(x).replace("D", "E").replace("d", "e"))
    if not math.isfinite(y):
        raise ValueError("non-finite value")
    return y


def get_num(point, aliases, name):
    d = {norm(k): v for k, v in point.items()}
    for alias in aliases:
        key = norm(alias)
        if key in d and str(d[key]).strip() != "":
            return finite_float(d[key])
    raise KeyError("Missing %s; available keys=%s" % (name, list(point)))


def physical(point):
    p = {
        k: get_num(point, aliases, k)
        for k, aliases in ALIASES.items()
        if k != "cba"
    }

    try:
        p["cba"] = get_num(point, ALIASES["cba"], "cba")
    except KeyError:
        # In this inverted scan beta-alpha lies near zero, hence the
        # positive cosine branch is the intended convention.
        p["cba"] = math.sqrt(max(0.0, 1.0 - p["sinba"] ** 2))

    if p["tanb"] <= 0:
        raise ValueError("tanb must be positive")

    if abs(p["sinba"] ** 2 + p["cba"] ** 2 - 1.0) > 5e-3:
        raise ValueError("sin(beta-alpha)^2 + cos(beta-alpha)^2 inconsistent")

    return p


def parse_stu(stdout_text):
    vals = []
    for k in "STU":
        m = STU_PAT[k].search(stdout_text)
        if not m:
            raise ValueError("Cannot parse %s from CalcPhys output" % k)
        vals.append(finite_float(m.group(1)))
    return tuple(vals)


# ============================================================================
# 3x3 correlated STU chi^2 without NumPy
# ============================================================================

def inverse_3x3(a):
    a00, a01, a02 = a[0]
    a10, a11, a12 = a[1]
    a20, a21, a22 = a[2]

    c00 = a11*a22 - a12*a21
    c01 = -(a10*a22 - a12*a20)
    c02 = a10*a21 - a11*a20

    c10 = -(a01*a22 - a02*a21)
    c11 = a00*a22 - a02*a20
    c12 = -(a00*a21 - a01*a20)

    c20 = a01*a12 - a02*a11
    c21 = -(a00*a12 - a02*a10)
    c22 = a00*a11 - a01*a10

    det = a00*c00 + a01*c01 + a02*c02

    if abs(det) < 1e-30:
        raise ValueError("STU covariance matrix is singular")

    # inverse = transpose(cofactor)/det
    return [
        [c00/det, c10/det, c20/det],
        [c01/det, c11/det, c21/det],
        [c02/det, c12/det, c22/det],
    ]


def make_stu_inverse(sigmas, rhos):
    ss, st, su = sigmas
    rst, rsu, rtu = rhos

    corr = [
        [1.0, rst, rsu],
        [rst, 1.0, rtu],
        [rsu, rtu, 1.0],
    ]

    sig = [ss, st, su]
    cov = [
        [sig[i] * sig[j] * corr[i][j] for j in range(3)]
        for i in range(3)
    ]

    return cov, inverse_3x3(cov)


def stu_chi2(S, T, U, center, invcov):
    d = [S-center[0], T-center[1], U-center[2]]
    return sum(
        d[i] * invcov[i][j] * d[j]
        for i in range(3)
        for j in range(3)
    )


# ============================================================================
# SLHA DECAY parser + HiggsTools dictionary
# Reuses the established direct Type-I HiggsTools construction.
# ============================================================================

def fnum(x):
    return float(x.replace("D", "E").replace("d", "e"))


def parse_decays(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    out = {}
    parent = None

    with path.open(errors="replace") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            t = line.split()

            if t[0].upper() == "DECAY":
                parent = int(t[1])
                out[parent] = {
                    "width": fnum(t[2]),
                    "channels": []
                }
                continue

            if t[0].upper() == "BLOCK":
                parent = None
                continue

            if parent is None or len(t) < 3:
                continue

            try:
                br = fnum(t[0])
                nda = int(t[1])
                daughters = tuple(map(int, t[2:2+nda]))

                if len(daughters) == nda:
                    out[parent]["channels"].append((br, daughters))
            except Exception:
                pass

    return out


PAIR = {
    frozenset((1, -1)): "dd",
    frozenset((2, -2)): "uu",
    frozenset((3, -3)): "ss",
    frozenset((4, -4)): "cc",
    frozenset((5, -5)): "bb",
    frozenset((6, -6)): "tt",
    frozenset((11, -11)): "ee",
    frozenset((13, -13)): "mumu",
    frozenset((15, -15)): "tautau",
    frozenset((21, 21)): "gg",
    frozenset((22, 22)): "gamgam",
    frozenset((23, 23)): "ZZ",
    frozenset((24, -24)): "WW",
    frozenset((22, 23)): "Zgam",
    frozenset((6, -5)): "tb",
    frozenset((-6, 5)): "tb",
    frozenset((4, -3)): "cs",
    frozenset((-4, 3)): "cs",
    frozenset((4, -5)): "cb",
    frozenset((-4, 5)): "cb",
    frozenset((15, 16)): "taunu",
    frozenset((-15, -16)): "taunu",
    frozenset((13, 14)): "munu",
    frozenset((-13, -14)): "munu",
    frozenset((11, 12)): "enu",
    frozenset((-11, -12)): "enu",
}

HIGGS = {25, 35, 36, 37, -37}


def map_decay(ds):
    if len(ds) != 2:
        return None

    a, b = ds

    pair = frozenset((a, b))
    if pair in PAIR:
        return (PAIR[pair],)

    if abs(a) in (12, 14, 16) and b == -a:
        return ("inv",)

    if a in HIGGS or b in HIGGS:
        return (str(a), str(b))

    return None


def add_scalar(data, pdg, k):
    for f in ("uu", "dd", "cc", "ss", "tt", "bb",
              "ee", "mumu", "tautau"):
        data["effc_%s_%s_s" % (pdg, f)] = k


def add_pseudo(data, pdg, cotb):
    for f in ("uu", "cc", "tt"):
        data["effc_%s_%s_p" % (pdg, f)] = cotb

    for f in ("dd", "ss", "bb", "ee", "mumu", "tautau"):
        data["effc_%s_%s_p" % (pdg, f)] = -cotb


def build_higgstools_dict(point, decays, cp_even, cp_odd,
                          charged=True, explicit=True, strict=False):
    p = physical(point)

    required = [25, 35, 36] + ([37] if charged else [])
    missing = [pdg for pdg in required if pdg not in decays]

    if missing:
        raise ValueError("Missing DECAY blocks for %s" % missing)

    d = {
        "m_25": p["mh"],
        "m_35": p["mH"],
        "m_36": p["mA"],

        "w_25": decays[25]["width"],
        "w_35": decays[35]["width"],
        "w_36": decays[36]["width"],

        "CP_25": cp_even,
        "CP_35": cp_even,
        "CP_36": cp_odd,
    }

    if charged:
        d.update({
            "m_37": p["mHp"],
            "w_37": decays[37]["width"]
        })

    s = p["sinba"]
    c = p["cba"]
    t = p["tanb"]

    # Type-I CP-even reduced fermion couplings:
    # hff = s_(b-a) + c_(b-a)/tanb
    # Hff = c_(b-a) - s_(b-a)/tanb
    add_scalar(d, 25, s + c/t)
    add_scalar(d, 35, c - s/t)

    # Type-I A couplings
    add_pseudo(d, 36, 1.0/t)

    d.update({
        "effc_25_WW": s,
        "effc_25_ZZ": s,
        "effc_35_WW": c,
        "effc_35_ZZ": c,
        "effc_36_WW": 0.0,
        "effc_36_ZZ": 0.0,
    })

    unmapped = []

    for parent in [25, 35, 36] + ([37] if charged else []):
        if parent in (25, 35, 36) and not explicit:
            continue

        for br, ds in decays[parent]["channels"]:
            if br <= 0:
                continue

            mapped = map_decay(ds)

            if mapped is None:
                unmapped.append({
                    "parent": parent,
                    "daughters": list(ds),
                    "br": br
                })

                if strict:
                    raise ValueError(
                        "Unmapped decay %s -> %s, BR=%s"
                        % (parent, ds, br)
                    )
                continue

            key = "br_%s_%s" % (
                parent,
                "_".join(mapped)
            )

            d[key] = d.get(key, 0.0) + br

    return d, unmapped


# ============================================================================
# Worker process state
# ============================================================================

_WORKER = {}


def init_worker(config):
    import Higgs.predictions as HP
    import Higgs.bounds as HB
    import Higgs.signals as HS
    import Higgs.tools.Input as HI

    tmp_root = Path(config["tmp_root"])
    tmp_root.mkdir(parents=True, exist_ok=True)

    _WORKER.update({
        "calcphys": config["calcphys"],
        "yukawa_type": config["yukawa_type"],
        "lambda6": config["lambda6"],
        "lambda7": config["lambda7"],
        "timeout": config["timeout"],

        "center": config["center"],
        "invcov": config["invcov"],
        "stu_cut": config["stu_cut"],

        "HP": HP,
        "HI": HI,
        "bounds": HB.Bounds(config["hb_data"]),
        "signals": HS.Signals(config["hs_data"]),
        "cp_even": int(HP.CP.even.value),
        "cp_odd": int(HP.CP.odd.value),

        "charged": config["charged"],
        "explicit_br": config["explicit_br"],
        "strict_decays": config["strict_decays"],
        "hs_chi2_max": config["hs_chi2_max"],

        "tmp_root": str(tmp_root),
    })


def process_one(task):
    row_number, point = task
    out = dict(point)

    pid = point.get(
        "point",
        point.get("point_id", row_number)
    )

    out["scan_row"] = row_number
    out["point_id"] = pid

    slha = Path(_WORKER["tmp_root"]) / (
        "p_%s_pid_%s_%s.slha"
        % (os.getpid(), str(pid), row_number)
    )

    out.update({
        "S": "",
        "T": "",
        "U": "",
        "chi2_STU": "",
        "STU_allowed": False,
        "HB_allowed": "",
        "HS_chi2": "",
        "HS_allowed": "",
        "combined_allowed": False,
        "status": "",
        "mapped_decay_count": "",
        "unmapped_decay_count": "",
        "error": "",
    })

    try:
        p = physical(point)

        cmd = [
            _WORKER["calcphys"],
            "%.17g" % p["mh"],
            "%.17g" % p["mH"],
            "%.17g" % p["mA"],
            "%.17g" % p["mHp"],
            "%.17g" % p["sinba"],
            "%.17g" % _WORKER["lambda6"],
            "%.17g" % _WORKER["lambda7"],
            "%.17g" % p["m12sq"],
            "%.17g" % p["tanb"],
            str(_WORKER["yukawa_type"]),
            str(slha),
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=_WORKER["timeout"],
        )

        stdout = proc.stdout or ""

        if proc.returncode != 0:
            raise RuntimeError(
                "CalcPhys exit %d: %s"
                % (proc.returncode, stdout[-1000:])
            )

        if not slha.is_file():
            raise RuntimeError("CalcPhys did not create SLHA file")

        S, T, U = parse_stu(stdout)

        chi2 = stu_chi2(
            S, T, U,
            _WORKER["center"],
            _WORKER["invcov"]
        )

        stu_ok = chi2 < _WORKER["stu_cut"]

        out.update({
            "S": S,
            "T": T,
            "U": U,
            "chi2_STU": chi2,
            "STU_allowed": stu_ok,
        })

        # ----------------------------------------------------
        # Early rejection: do not run HiggsTools for STU failures
        # ----------------------------------------------------
        if not stu_ok:
            out["status"] = "STU_REJECTED"
            return "stu_rejected", out

        # ----------------------------------------------------
        # HiggsTools only for STU survivors
        # ----------------------------------------------------
        decays = parse_decays(slha)

        pred_dict, unmapped = build_higgstools_dict(
            point,
            decays,
            _WORKER["cp_even"],
            _WORKER["cp_odd"],
            _WORKER["charged"],
            _WORKER["explicit_br"],
            _WORKER["strict_decays"],
        )

        out["mapped_decay_count"] = sum(
            1 for k in pred_dict if k.startswith("br_")
        )
        out["unmapped_decay_count"] = len(unmapped)

        # HiggsTools BsmParticle IDs are strings in the installed
        # Python interface.  Passing integer PDG IDs causes
        #
        #   TypeError: BsmParticle(id: str, ...)
        #
        # inside predictionsFromDict().  The dictionary keys remain
        # m_25, w_25, br_25_..., etc.; only the ID lists supplied to
        # predictionsFromDict must be strings.
        neutral_ids = ["25", "35", "36"]
        charged_ids = ["37"] if _WORKER["charged"] else []

        prediction = _WORKER["HI"].predictionsFromDict(
            pred_dict,
            neutral_ids,
            charged_ids,
            [],
            useExplicitBr=_WORKER["explicit_br"],
            calcggH=True,
            calcHgamgam=True,
        )

        hb_allowed = bool(
            _WORKER["bounds"](prediction).allowed
        )

        hs_chi2 = float(
            _WORKER["signals"](prediction)
        )

        if not math.isfinite(hs_chi2):
            raise ValueError("Non-finite HiggsSignals chi2")

        hs_cut = _WORKER["hs_chi2_max"]

        if hs_cut is None:
            hs_allowed = None
            combined = hb_allowed
        else:
            hs_allowed = hs_chi2 < hs_cut
            combined = hb_allowed and hs_allowed

        out.update({
            "HB_allowed": hb_allowed,
            "HS_chi2": hs_chi2,
            "HS_allowed": "" if hs_allowed is None else hs_allowed,
            "combined_allowed": combined,
        })

        if not hb_allowed:
            out["status"] = "HB_REJECTED"
            return "hb_rejected", out

        if hs_allowed is False:
            out["status"] = "HS_REJECTED"
            return "hs_rejected", out

        if hs_cut is None:
            out["status"] = "HB_ALLOWED_HS_RECORDED"
        else:
            out["status"] = "ALLOWED"

        return "allowed", out

    except subprocess.TimeoutExpired:
        out["status"] = "FAILED_2HDMC_TIMEOUT"
        out["error"] = "CalcPhys timeout after %s s" % _WORKER["timeout"]
        return "failed", out

    except Exception as e:
        out["status"] = "FAILED"
        out["error"] = "%s: %s" % (type(e).__name__, e)
        return "failed", out

    finally:
        try:
            if slha.exists():
                slha.unlink()
        except Exception:
            pass


# ============================================================================
# Streaming CSV helpers
# ============================================================================

RESULT_EXTRA_FIELDS = [
    "scan_row",
    "point_id",
    "S",
    "T",
    "U",
    "chi2_STU",
    "STU_allowed",
    "HB_allowed",
    "HS_chi2",
    "HS_allowed",
    "combined_allowed",
    "status",
    "mapped_decay_count",
    "unmapped_decay_count",
    "error",
]


def open_output_writer(path, fieldnames, resume):
    mode = "a" if resume and path.exists() and path.stat().st_size > 0 else "w"
    f = path.open(mode, newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

    if mode == "w":
        w.writeheader()
        f.flush()

    return f, w


def count_data_rows(path):
    # Only used for optional validation, not for normal processing.
    with path.open() as f:
        return max(0, sum(1 for _ in f) - 1)


def read_chunk(reader, n):
    rows = []
    for _ in range(n):
        try:
            rows.append(next(reader))
        except StopIteration:
            break
    return rows


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", default=DEF_INPUT)
    p.add_argument("--calcphys", default=DEF_CALCPHYS)
    p.add_argument("--hb-data", default=DEF_HB_DATA)
    p.add_argument("--hs-data", default=DEF_HS_DATA)
    p.add_argument("--output-dir", default="constraints_2HDMC_HiggsTools")

    p.add_argument("--ncpu", type=int, default=8)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="rows processed per checkpointed wave"
    )
    p.add_argument(
        "--pool-chunksize",
        type=int,
        default=1,
        help="multiprocessing imap chunk size"
    )
    p.add_argument("--max-points", type=int)
    p.add_argument("--start-row", type=int, default=0)

    p.add_argument("--yukawa-type", type=int, default=1)
    p.add_argument("--lambda6", type=float, default=0.0)
    p.add_argument("--lambda7", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=60.0)

    # Existing correlated STU fit defaults
    p.add_argument("--chi2-cut", type=float, default=7.815)
    p.add_argument("--s0", type=float, default=-0.04)
    p.add_argument("--t0", type=float, default=0.01)
    p.add_argument("--u0", type=float, default=-0.01)
    p.add_argument("--sigma-s", type=float, default=0.10)
    p.add_argument("--sigma-t", type=float, default=0.12)
    p.add_argument("--sigma-u", type=float, default=0.09)
    p.add_argument("--rho-st", type=float, default=0.92)
    p.add_argument("--rho-su", type=float, default=-0.80)
    p.add_argument("--rho-tu", type=float, default=-0.93)

    p.add_argument(
        "--hs-chi2-max",
        type=float,
        default=None,
        help=(
            "Optional HiggsSignals chi2 rejection threshold. "
            "If omitted, HS chi2 is recorded but not used to reject."
        )
    )

    p.add_argument("--no-charged-higgs", action="store_true")
    p.add_argument("--no-explicit-br", action="store_true")
    p.add_argument("--strict-decays", action="store_true")

    p.add_argument(
        "--save-all-results",
        action="store_true",
        help="also save one all-results row for every input point"
    )

    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore/remove an existing checkpoint and start over"
    )

    return p.parse_args()


def main():
    a = parse_args()

    input_path = Path(a.input).expanduser().resolve()
    calcphys = Path(a.calcphys).expanduser().resolve()
    hb_data = Path(a.hb_data).expanduser().resolve()
    hs_data = Path(a.hs_data).expanduser().resolve()
    outdir = Path(a.output_dir).expanduser().resolve()

    if not input_path.is_file():
        sys.exit("Input CSV not found: %s" % input_path)

    if not calcphys.is_file():
        sys.exit("CalcPhys not found: %s" % calcphys)

    if not hb_data.is_dir():
        sys.exit("HiggsBounds dataset not found: %s" % hb_data)

    if not hs_data.is_dir():
        sys.exit("HiggsSignals dataset not found: %s" % hs_data)

    outdir.mkdir(parents=True, exist_ok=True)

    tmp_root = outdir / "tmp_slha"
    tmp_root.mkdir(exist_ok=True)

    checkpoint = outdir / "checkpoint.json"

    final_allowed_path = outdir / "Allowed_after_STU_HB_HS.csv"
    stu_allowed_path = outdir / "Allowed_after_STU.csv"
    failures_path = outdir / "Failed_constraints.csv"
    all_results_path = outdir / "All_constraint_results.csv"

    if a.fresh:
        if checkpoint.exists():
            checkpoint.unlink()

        for path in (
            final_allowed_path,
            stu_allowed_path,
            failures_path,
            all_results_path,
        ):
            if path.exists():
                path.unlink()

        if tmp_root.exists():
            shutil.rmtree(str(tmp_root), ignore_errors=True)
            tmp_root.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # STU covariance
    # --------------------------------------------------------
    cov, invcov = make_stu_inverse(
        (a.sigma_s, a.sigma_t, a.sigma_u),
        (a.rho_st, a.rho_su, a.rho_tu),
    )

    center = [a.s0, a.t0, a.u0]

    # --------------------------------------------------------
    # Checkpoint/resume
    # next_row counts input DATA rows, zero-based.
    # --------------------------------------------------------
    state = {
        "next_row": a.start_row,
        "processed": 0,
        "stu_allowed": 0,
        "stu_rejected": 0,
        "hb_rejected": 0,
        "hs_rejected": 0,
        "allowed": 0,
        "failed": 0,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    resume = False

    if checkpoint.exists() and not a.fresh:
        saved = json.loads(checkpoint.read_text())

        # Protect against accidentally resuming a different input.
        if saved.get("input") and saved["input"] != str(input_path):
            sys.exit(
                "Checkpoint belongs to different input: %s"
                % saved["input"]
            )

        state.update(saved)
        resume = True

    start_row = int(state["next_row"])

    if a.start_row and resume and a.start_row != start_row:
        print(
            "NOTE: checkpoint next_row=%d overrides --start-row=%d"
            % (start_row, a.start_row)
        )

    # --------------------------------------------------------
    # Read header first and construct output schema
    # --------------------------------------------------------
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        input_fields = list(reader.fieldnames or [])

    output_fields = list(input_fields)

    for k in RESULT_EXTRA_FIELDS:
        if k not in output_fields:
            output_fields.append(k)

    allowed_file, allowed_writer = open_output_writer(
        final_allowed_path, output_fields, resume
    )

    stu_file, stu_writer = open_output_writer(
        stu_allowed_path, output_fields, resume
    )

    fail_file, fail_writer = open_output_writer(
        failures_path, output_fields, resume
    )

    all_file = None
    all_writer = None

    if a.save_all_results:
        all_file, all_writer = open_output_writer(
            all_results_path, output_fields, resume
        )

    # --------------------------------------------------------
    # Worker configuration
    # --------------------------------------------------------
    config = {
        "calcphys": str(calcphys),
        "yukawa_type": a.yukawa_type,
        "lambda6": a.lambda6,
        "lambda7": a.lambda7,
        "timeout": a.timeout,

        "center": center,
        "invcov": invcov,
        "stu_cut": a.chi2_cut,

        "hb_data": str(hb_data),
        "hs_data": str(hs_data),
        "charged": not a.no_charged_higgs,
        "explicit_br": not a.no_explicit_br,
        "strict_decays": a.strict_decays,
        "hs_chi2_max": a.hs_chi2_max,

        "tmp_root": str(tmp_root),
    }

    ncpu = max(1, min(a.ncpu, mp.cpu_count()))
    chunk_size = max(1, a.chunk_size)
    pool_chunksize = max(1, a.pool_chunksize)

    print("\n============================================================")
    print("Integrated 2HDMC + STU + HiggsTools scan")
    print("============================================================")
    print("Input             :", input_path)
    print("2HDMC CalcPhys    :", calcphys)
    print("HiggsBounds data  :", hb_data)
    print("HiggsSignals data :", hs_data)
    print("Yukawa type       :", a.yukawa_type)
    print("CPUs              :", ncpu)
    print("Checkpoint chunk  :", chunk_size)
    print("Starting data row :", start_row)
    print("STU chi2 cut      :", a.chi2_cut)
    print("STU center        :", center)
    print("HS chi2 cut       :", a.hs_chi2_max)
    if a.hs_chi2_max is None:
        print("HS policy         : record chi2; do NOT reject by HS")
    else:
        print("HS policy         : require chi2 < %.6g" % a.hs_chi2_max)
    print("============================================================\n")

    pool = mp.Pool(
        processes=ncpu,
        initializer=init_worker,
        initargs=(config,),
    )

    run_processed = 0
    max_points = a.max_points

    try:
        with input_path.open(newline="") as f:
            reader = csv.DictReader(f)

            # Skip already checkpointed rows.
            for _ in range(start_row):
                try:
                    next(reader)
                except StopIteration:
                    break

            row_index = start_row

            while True:
                if max_points is not None:
                    remain = max_points - run_processed
                    if remain <= 0:
                        break
                    this_chunk_n = min(chunk_size, remain)
                else:
                    this_chunk_n = chunk_size

                rows = read_chunk(reader, this_chunk_n)

                if not rows:
                    break

                tasks = [
                    (row_index + i, row)
                    for i, row in enumerate(rows)
                ]

                chunk_counts = {
                    "stu_allowed": 0,
                    "stu_rejected": 0,
                    "hb_rejected": 0,
                    "hs_rejected": 0,
                    "allowed": 0,
                    "failed": 0,
                }

                for status, result in pool.imap_unordered(
                    process_one,
                    tasks,
                    chunksize=pool_chunksize,
                ):
                    # STU survivor audit file includes everything that passed STU,
                    # whether HB/HS later rejects it or not.
                    if result.get("STU_allowed") is True:
                        stu_writer.writerow(result)
                        chunk_counts["stu_allowed"] += 1

                    if status == "stu_rejected":
                        chunk_counts["stu_rejected"] += 1

                    elif status == "hb_rejected":
                        chunk_counts["hb_rejected"] += 1

                    elif status == "hs_rejected":
                        chunk_counts["hs_rejected"] += 1

                    elif status == "allowed":
                        allowed_writer.writerow(result)
                        chunk_counts["allowed"] += 1

                    else:
                        fail_writer.writerow(result)
                        chunk_counts["failed"] += 1

                    if all_writer is not None:
                        all_writer.writerow(result)

                # Flush COMPLETE chunk before checkpoint advances.
                allowed_file.flush()
                stu_file.flush()
                fail_file.flush()

                if all_file is not None:
                    all_file.flush()

                processed_here = len(rows)
                row_index += processed_here
                run_processed += processed_here

                state["next_row"] = row_index
                state["processed"] = int(state.get("processed", 0)) + processed_here

                for k in chunk_counts:
                    state[k] = int(state.get(k, 0)) + chunk_counts[k]

                state["input"] = str(input_path)
                state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                state["hs_chi2_max"] = a.hs_chi2_max
                state["stu_chi2_cut"] = a.chi2_cut

                tmp_checkpoint = Path(str(checkpoint) + ".tmp")
                tmp_checkpoint.write_text(
                    json.dumps(state, indent=2)
                )
                tmp_checkpoint.replace(checkpoint)

                print(
                    "rows %d-%d | "
                    "STU pass=%d reject=%d | "
                    "HB reject=%d | HS reject=%d | "
                    "FINAL allowed=%d | failed=%d | "
                    "TOTAL processed=%d"
                    % (
                        row_index - processed_here,
                        row_index - 1,
                        chunk_counts["stu_allowed"],
                        chunk_counts["stu_rejected"],
                        chunk_counts["hb_rejected"],
                        chunk_counts["hs_rejected"],
                        chunk_counts["allowed"],
                        chunk_counts["failed"],
                        state["processed"],
                    )
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nInterrupted. Closing worker pool...")
        pool.terminate()
        pool.join()
        print(
            "Checkpoint preserved at %s. "
            "Rerun the SAME command to resume." % checkpoint
        )
        return

    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass

        allowed_file.close()
        stu_file.close()
        fail_file.close()

        if all_file is not None:
            all_file.close()

        # Remove any stale temporary files left by abnormal worker exits.
        if tmp_root.exists():
            for p in tmp_root.glob("*.slha"):
                try:
                    p.unlink()
                except Exception:
                    pass

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary_path = outdir / "summary.json"

    summary = dict(state)
    summary.update({
        "input": str(input_path),
        "calcphys": str(calcphys),
        "hb_data": str(hb_data),
        "hs_data": str(hs_data),
        "ncpu": ncpu,
        "chunk_size": chunk_size,
        "stu_center": center,
        "stu_covariance": cov,
        "stu_chi2_cut": a.chi2_cut,
        "hs_chi2_max": a.hs_chi2_max,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    summary_path.write_text(
        json.dumps(summary, indent=2)
    )

    print("\n============================================================")
    print("RUN COMPLETE / STOPPED AT REQUESTED LIMIT")
    print("============================================================")
    print("Processed total :", state["processed"])
    print("STU allowed     :", state["stu_allowed"])
    print("STU rejected    :", state["stu_rejected"])
    print("HB rejected     :", state["hb_rejected"])
    print("HS rejected     :", state["hs_rejected"])
    print("Final allowed   :", state["allowed"])
    print("Failed          :", state["failed"])
    print("Checkpoint      :", checkpoint)
    print("Summary         :", summary_path)
    print("Final CSV       :", final_allowed_path)
    print("STU-pass CSV    :", stu_allowed_path)
    print("============================================================")


if __name__ == "__main__":
    mp.freeze_support()
    main()
