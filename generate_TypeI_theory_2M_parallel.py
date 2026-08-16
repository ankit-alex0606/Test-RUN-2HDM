#!/usr/bin/env python3
"""
Parallel, streaming generator for inverted Type-I 2HDM theory-allowed points.

Designed for large production scans such as 2,000,000 ACCEPTED points.

Applied BEFORE writing a point:
  1) Bounded-from-below (BFB)
  2) Perturbativity: |lambda_i| < 4*pi
  3) Tree-level scalar-scattering unitarity: |eigenvalue| < 8*pi

Model assumptions:
  * CP-conserving 2HDM
  * softly broken Z2
  * lambda6 = lambda7 = 0
  * inverted scenario: H = 125 GeV, h < 125 GeV
  * Type-I physical scan ranges used in the previous workflow

Key features:
  * Uses multiple CPU cores with multiprocessing.
  * Each worker writes directly to its own CSV -> low RAM usage.
  * Each worker has its own checkpoint -> resumable.
  * Exact total accepted target is divided across workers.
  * Optional final merge into one CSV.
  * Python 3.6 compatible (no pandas/numpy required).

Example:
  python3 generate_TypeI_theory_2M_parallel.py \
      --target 2000000 \
      --parallel 8 \
      --seed 12345 \
      --output-dir theory_2M \
      --merge
"""

from __future__ import print_function

import argparse
import base64
import csv
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path


# ============================================================================
# Physical scan ranges
# ============================================================================

MH_FIXED = 125.0

MHC_MIN, MHC_MAX = 102.7, 351.4
MA_MIN,  MA_MAX  = 63.2, 356.4
Mh_MIN,  Mh_MAX  = 62.5, 114.4

SINBMA_MIN, SINBMA_MAX = -0.12, 0.30
MU212_MIN,  MU212_MAX  = 29.0, 2775.0
TANB_MIN,   TANB_MAX   = 2.5, 45.4

VEV = 246.0
V2 = VEV * VEV

PERT_LIMIT = 4.0 * math.pi
UNITARITY_LIMIT = 8.0 * math.pi


# ============================================================================
# Output schema
# ============================================================================

OUTPUT_FIELDS = [
    "point",
    "worker",
    "worker_local_index",
    "Mh",
    "MH",
    "MA",
    "MHc",
    "tanb",
    "sinbma",
    "mu212",
    "alpha_mix",
    "beta_mix",
    "lam1",
    "lam2",
    "lam3",
    "lam4",
    "lam5",
    "m11sq",
    "m22sq",
    "BFB_allowed",
    "perturbative_allowed",
    "unitarity_allowed",
    "max_abs_unitarity_eigenvalue",
]


# ============================================================================
# Physical input -> quartic couplings
# ============================================================================

def physical_to_lambdas(Mh, MH, MA, MHc, tanb, sinbma, mu212):
    """
    mu212 = m12^2 [GeV^2]

    Convention matches the PhaseTracer Type-I physical-parameter conversion
    used in the previous scan scripts.
    """

    if tanb <= 0.0:
        raise ValueError("tanb <= 0")

    if abs(sinbma) > 1.0:
        raise ValueError("|sin(beta-alpha)| > 1")

    beta = math.atan(tanb)
    alpha = beta - math.asin(sinbma)

    sb = math.sin(beta)
    cb = math.cos(beta)
    sa = math.sin(alpha)
    ca = math.cos(alpha)

    if abs(sb * cb) < 1.0e-14:
        raise ValueError("sin(beta)*cos(beta) too small")

    mh2 = Mh * Mh
    mH2 = MH * MH
    mA2 = MA * MA
    mC2 = MHc * MHc

    lam1 = (
        mH2 * ca * ca
        + mh2 * sa * sa
        - mu212 * tanb
    ) / (V2 * cb * cb)

    lam2 = (
        mH2 * sa * sa
        + mh2 * ca * ca
        - mu212 / tanb
    ) / (V2 * sb * sb)

    lam3 = (
        (mH2 - mh2) * sa * ca / (sb * cb)
        + 2.0 * mC2
        - mu212 / (sb * cb)
    ) / V2

    lam4 = (
        mA2
        - 2.0 * mC2
        + mu212 / (sb * cb)
    ) / V2

    lam5 = (
        mu212 / (sb * cb)
        - mA2
    ) / V2

    return alpha, beta, lam1, lam2, lam3, lam4, lam5


# ============================================================================
# Theoretical constraints
# ============================================================================

def passes_bfb(l1, l2, l3, l4, l5):
    """
    Tree-level BFB conditions for CP-conserving Z2-symmetric quartic sector
    with lambda6=lambda7=0.
    """
    if l1 <= 0.0 or l2 <= 0.0:
        return False

    root = math.sqrt(l1 * l2)

    return (
        l3 > -root
        and
        l3 + l4 - abs(l5) > -root
    )


def passes_perturbativity(l1, l2, l3, l4, l5):
    return all(
        abs(x) < PERT_LIMIT
        for x in (l1, l2, l3, l4, l5)
    )


def unitarity_eigenvalues(l1, l2, l3, l4, l5):
    """
    Standard tree-level scalar-scalar scattering combinations for
    CP-conserving 2HDM with lambda6=lambda7=0.
    """

    root_a = math.sqrt(
        2.25 * (l1 - l2) ** 2
        + (2.0 * l3 + l4) ** 2
    )

    root_b = math.sqrt(
        0.25 * (l1 - l2) ** 2
        + l4 ** 2
    )

    root_c = math.sqrt(
        0.25 * (l1 - l2) ** 2
        + l5 ** 2
    )

    return [
        1.5 * (l1 + l2) + root_a,
        1.5 * (l1 + l2) - root_a,

        0.5 * (l1 + l2) + root_b,
        0.5 * (l1 + l2) - root_b,

        0.5 * (l1 + l2) + root_c,
        0.5 * (l1 + l2) - root_c,

        l3 + 2.0 * l4 - 3.0 * l5,
        l3 - l5,

        l3 + 2.0 * l4 + 3.0 * l5,
        l3 + l5,

        l3 + l4,
        l3 + l4,
        l3 - l4,
    ]


def passes_unitarity(l1, l2, l3, l4, l5):
    eig = unitarity_eigenvalues(l1, l2, l3, l4, l5)

    max_abs = max(abs(x) for x in eig)

    return (
        all(abs(x) < UNITARITY_LIMIT for x in eig),
        max_abs,
    )


# ============================================================================
# Tadpole-derived m11^2, m22^2
# ============================================================================

def quadratic_masses(tanb, mu212, l1, l2, l3, l4, l5):
    beta = math.atan(tanb)

    sb = math.sin(beta)
    cb = math.cos(beta)

    lam345 = l3 + l4 + l5

    m11sq = (
        mu212 * tanb
        - 0.5 * V2 * (
            l1 * cb * cb
            + lam345 * sb * sb
        )
    )

    m22sq = (
        mu212 / tanb
        - 0.5 * V2 * (
            l2 * sb * sb
            + lam345 * cb * cb
        )
    )

    return m11sq, m22sq


# ============================================================================
# RNG state serialization for resumable workers
# ============================================================================

def encode_rng_state(state):
    raw = pickle.dumps(state, protocol=2)
    return base64.b64encode(raw).decode("ascii")


def decode_rng_state(text):
    raw = base64.b64decode(text.encode("ascii"))
    return pickle.loads(raw)


# ============================================================================
# Worker
# ============================================================================

def worker_generate(
    worker_id,
    worker_target,
    global_start_index,
    base_seed,
    outdir_str,
    checkpoint_every,
    progress_every,
):
    outdir = Path(outdir_str)

    csv_path = outdir / (
        "Allowed_2HDM_theory_worker_%02d.csv"
        % worker_id
    )

    state_path = outdir / (
        "worker_%02d_state.json"
        % worker_id
    )

    rng = random.Random(
        base_seed + 1000003 * worker_id
    )

    accepted = 0
    attempts = 0

    fail_bfb = 0
    fail_pert = 0
    fail_unitarity = 0
    invalid = 0

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if state_path.exists():

        state = json.loads(
            state_path.read_text()
        )

        accepted = int(
            state.get("accepted", 0)
        )

        attempts = int(
            state.get("attempts", 0)
        )

        fail_bfb = int(
            state.get("fail_bfb", 0)
        )

        fail_pert = int(
            state.get("fail_pert", 0)
        )

        fail_unitarity = int(
            state.get("fail_unitarity", 0)
        )

        invalid = int(
            state.get("invalid", 0)
        )

        if state.get("rng_state"):
            rng.setstate(
                decode_rng_state(
                    state["rng_state"]
                )
            )

    # --------------------------------------------------------
    # Open CSV in append mode
    # --------------------------------------------------------

    file_exists = (
        csv_path.exists()
        and
        csv_path.stat().st_size > 0
    )

    with csv_path.open(
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_FIELDS,
        )

        if not file_exists:
            writer.writeheader()
            f.flush()

        last_checkpoint_accepted = accepted
        last_progress_accepted = accepted

        while accepted < worker_target:

            attempts += 1

            # ------------------------------------------------
            # Random physical point
            # ------------------------------------------------

            Mh = rng.uniform(
                Mh_MIN,
                Mh_MAX
            )

            MH = MH_FIXED

            MA = rng.uniform(
                MA_MIN,
                MA_MAX
            )

            MHc = rng.uniform(
                MHC_MIN,
                MHC_MAX
            )

            tanb = rng.uniform(
                TANB_MIN,
                TANB_MAX
            )

            sinbma = rng.uniform(
                SINBMA_MIN,
                SINBMA_MAX
            )

            mu212 = rng.uniform(
                MU212_MIN,
                MU212_MAX
            )

            try:

                (
                    alpha,
                    beta,
                    l1,
                    l2,
                    l3,
                    l4,
                    l5,
                ) = physical_to_lambdas(
                    Mh,
                    MH,
                    MA,
                    MHc,
                    tanb,
                    sinbma,
                    mu212,
                )

            except Exception:

                invalid += 1
                continue

            couplings = (
                l1,
                l2,
                l3,
                l4,
                l5,
            )

            if not all(
                math.isfinite(x)
                for x in couplings
            ):

                invalid += 1
                continue

            # ------------------------------------------------
            # BFB
            # ------------------------------------------------

            if not passes_bfb(
                *couplings
            ):

                fail_bfb += 1
                continue

            # ------------------------------------------------
            # Perturbativity
            # ------------------------------------------------

            if not passes_perturbativity(
                *couplings
            ):

                fail_pert += 1
                continue

            # ------------------------------------------------
            # Unitarity
            # ------------------------------------------------

            ok_unitarity, max_unitarity = (
                passes_unitarity(
                    *couplings
                )
            )

            if not ok_unitarity:

                fail_unitarity += 1
                continue

            # ------------------------------------------------
            # Quadratic masses
            # ------------------------------------------------

            m11sq, m22sq = quadratic_masses(
                tanb,
                mu212,
                l1,
                l2,
                l3,
                l4,
                l5,
            )

            # ------------------------------------------------
            # Accepted -> immediately stream to CSV
            # ------------------------------------------------

            local_index = accepted

            global_point = (
                global_start_index
                +
                local_index
            )

            row = {
                "point": global_point,
                "worker": worker_id,
                "worker_local_index": local_index,

                "Mh": Mh,
                "MH": MH,
                "MA": MA,
                "MHc": MHc,

                "tanb": tanb,
                "sinbma": sinbma,
                "mu212": mu212,

                "alpha_mix": alpha,
                "beta_mix": beta,

                "lam1": l1,
                "lam2": l2,
                "lam3": l3,
                "lam4": l4,
                "lam5": l5,

                "m11sq": m11sq,
                "m22sq": m22sq,

                "BFB_allowed": 1,
                "perturbative_allowed": 1,
                "unitarity_allowed": 1,

                "max_abs_unitarity_eigenvalue": (
                    max_unitarity
                ),
            }

            writer.writerow(row)

            accepted += 1

            # ------------------------------------------------
            # Periodic flush + checkpoint
            # ------------------------------------------------

            if (
                accepted - last_checkpoint_accepted
                >= checkpoint_every
                or
                accepted == worker_target
            ):

                f.flush()

                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass

                state = {
                    "worker": worker_id,
                    "target": worker_target,
                    "accepted": accepted,
                    "attempts": attempts,

                    "fail_bfb": fail_bfb,
                    "fail_pert": fail_pert,
                    "fail_unitarity": fail_unitarity,
                    "invalid": invalid,

                    "rng_state": encode_rng_state(
                        rng.getstate()
                    ),

                    "updated": time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }

                tmp_state = Path(
                    str(state_path) + ".tmp"
                )

                tmp_state.write_text(
                    json.dumps(
                        state,
                        indent=2,
                    )
                )

                tmp_state.replace(
                    state_path
                )

                last_checkpoint_accepted = accepted

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                accepted - last_progress_accepted
                >= progress_every
                or
                accepted == worker_target
            ):

                rate = (
                    float(accepted)
                    / float(attempts)
                    if attempts
                    else 0.0
                )

                print(
                    "[worker %d] accepted %d/%d | "
                    "attempts %d | rate %.4f"
                    %
                    (
                        worker_id,
                        accepted,
                        worker_target,
                        attempts,
                        rate,
                    )
                )

                sys.stdout.flush()

                last_progress_accepted = accepted

    # --------------------------------------------------------
    # Worker summary
    # --------------------------------------------------------

    return {
        "worker": worker_id,
        "target": worker_target,
        "accepted": accepted,
        "attempts": attempts,
        "fail_bfb": fail_bfb,
        "fail_pert": fail_pert,
        "fail_unitarity": fail_unitarity,
        "invalid": invalid,
        "csv": str(csv_path),
        "state": str(state_path),
    }


# ============================================================================
# Merge worker CSVs
# ============================================================================

def merge_worker_csvs(outdir, nworkers, target):

    merged = outdir / (
        "Allowed_2HDM_theory_unitarity_%d.csv"
        % target
    )

    with merged.open(
        "w",
        newline=""
    ) as fout:

        writer = csv.DictWriter(
            fout,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()

        for wid in range(nworkers):

            part = outdir / (
                "Allowed_2HDM_theory_worker_%02d.csv"
                % wid
            )

            if not part.exists():
                raise RuntimeError(
                    "Missing worker file: %s"
                    % part
                )

            with part.open(
                newline=""
            ) as fin:

                reader = csv.DictReader(
                    fin
                )

                for row in reader:
                    writer.writerow(row)

    return merged


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=int,
        default=2000000,
        help=(
            "TOTAL accepted points wanted "
            "across all workers"
        ),
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help=(
            "number of generator worker processes"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--output-dir",
        default="theory_2M",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help=(
            "accepted points between worker checkpoints"
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help=(
            "accepted points between worker progress messages"
        ),
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "merge worker CSVs into one CSV after completion"
        ),
    )

    args = parser.parse_args()

    if args.target <= 0:
        raise SystemExit(
            "--target must be > 0"
        )

    if args.parallel <= 0:
        raise SystemExit(
            "--parallel must be > 0"
        )

    outdir = Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Exact target division among workers
    # --------------------------------------------------------

    base = (
        args.target
        // args.parallel
    )

    remainder = (
        args.target
        % args.parallel
    )

    worker_targets = []

    for wid in range(
        args.parallel
    ):

        n = base

        if wid < remainder:
            n += 1

        worker_targets.append(
            n
        )

    # Global ID start for each worker
    global_starts = []

    running = 0

    for n in worker_targets:

        global_starts.append(
            running
        )

        running += n

    print(
        "\n=============================================="
    )

    print(
        "Parallel inverted Type-I theory generation"
    )

    print(
        "=============================================="
    )

    print(
        "Total accepted target :",
        args.target,
    )

    print(
        "Workers               :",
        args.parallel,
    )

    print(
        "Worker targets        :",
        worker_targets,
    )

    print(
        "BFB                   : enabled"
    )

    print(
        "Perturbativity        : |lambda_i| < 4*pi"
    )

    print(
        "Unitarity             : |eigenvalue| < 8*pi"
    )

    print(
        "Output directory      :",
        outdir,
    )

    print(
        "Checkpoint every      :",
        args.checkpoint_every,
        "accepted/worker",
    )

    print(
        "==============================================\n"
    )

    # --------------------------------------------------------
    # Multiprocessing
    # --------------------------------------------------------

    jobs = []

    pool = mp.Pool(
        processes=args.parallel
    )

    try:

        for wid in range(
            args.parallel
        ):

            job = pool.apply_async(
                worker_generate,
                (
                    wid,
                    worker_targets[wid],
                    global_starts[wid],
                    args.seed,
                    str(outdir),
                    args.checkpoint_every,
                    args.progress_every,
                ),
            )

            jobs.append(job)

        pool.close()

        summaries = [
            job.get()
            for job in jobs
        ]

        pool.join()

    except KeyboardInterrupt:

        print(
            "\nInterrupted by user."
        )

        print(
            "Terminating workers..."
        )

        pool.terminate()
        pool.join()

        print(
            "Re-run the SAME command to resume "
            "from worker checkpoints."
        )

        return

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_accepted = sum(
        s["accepted"]
        for s in summaries
    )

    total_attempts = sum(
        s["attempts"]
        for s in summaries
    )

    total_bfb = sum(
        s["fail_bfb"]
        for s in summaries
    )

    total_pert = sum(
        s["fail_pert"]
        for s in summaries
    )

    total_unit = sum(
        s["fail_unitarity"]
        for s in summaries
    )

    print(
        "\n=============================================="
    )

    print(
        "GENERATION COMPLETE"
    )

    print(
        "=============================================="
    )

    print(
        "Accepted total         :",
        total_accepted,
    )

    print(
        "Requested total        :",
        args.target,
    )

    print(
        "Random attempts        :",
        total_attempts,
    )

    print(
        "BFB rejected           :",
        total_bfb,
    )

    print(
        "Perturbativity rejected:",
        total_pert,
    )

    print(
        "Unitarity rejected     :",
        total_unit,
    )

    if total_attempts:

        print(
            "Overall acceptance rate:",
            float(total_accepted)
            / float(total_attempts),
        )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # Write run manifest
    # --------------------------------------------------------

    manifest = {
        "target": args.target,
        "parallel": args.parallel,
        "seed": args.seed,
        "accepted": total_accepted,
        "attempts": total_attempts,
        "worker_targets": worker_targets,
        "summaries": summaries,
        "finished": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    manifest_path = (
        outdir
        / "generation_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print(
        "Manifest:",
        manifest_path,
    )

    # --------------------------------------------------------
    # Optional merge
    # --------------------------------------------------------

    if args.merge:

        print(
            "\nMerging worker CSV files..."
        )

        merged = merge_worker_csvs(
            outdir,
            args.parallel,
            args.target,
        )

        print(
            "Merged CSV:",
            merged,
        )

    else:

        print(
            "\nWorker CSV files were left separate."
        )

        print(
            "Use --merge if you want one final CSV."
        )


if __name__ == "__main__":
    main()
