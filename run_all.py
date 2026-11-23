# -*- coding: utf-8 -*-
"""
run_all.py -- reproduce all results and figures of
"Artificial intelligence turns investment in people into cooperation with the future"
(C. R. Klein and R. Klein)

Usage:
    python run_all.py            # full run (about 1-2 hours on a laptop)
    python run_all.py --quick    # fast test run with fewer seeds (results not for the paper)
    python run_all.py --figures  # only redraw the figures from the stored results

In Spyder: open this file and press F5. On Windows: double-click run_all.bat.
"""
import os
import sys
import time
import platform
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

STEPS = [
    ("schritt1_ohtsuki", "Step 1: cooperation conditions on the model networks (Ohtsuki, Allen et al.)"),
    ("schritt2_vierfeldertafel", "Step 2: task-level game, human vs AI"),
    ("schritt3_zukunft", "Step 3: cooperating with the future (variants, hysteresis, institutions)"),
    ("schritt4_regel", "Step 4: the rule, binding agreements, robustness"),
    ("schritt5_versaeumnis_lhs", "Step 5: single-lapse test and Latin hypercube"),
    ("schritt6_review", "Step 6: mechanism and robustness (Supplementary Fig. 2)"),
    ("schritt7_review2", "Step 7: rewiring, mapping, replicates and network size (Supplementary Fig. 3)"),
]


def main():
    quick = "--quick" in sys.argv
    figures_only = "--figures" in sys.argv
    t0 = time.time()
    log = ["run_all | %s | Python %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                               sys.version.split()[0], platform.platform())]
    import numpy, pandas, matplotlib, networkx
    log.append("numpy %s | pandas %s | matplotlib %s | networkx %s" % (
        numpy.__version__, pandas.__version__, matplotlib.__version__, networkx.__version__))
    if not figures_only:
        for mod_name, label in STEPS:
            print("\n" + "=" * 72 + "\n  " + label + "\n" + "=" * 72, flush=True)
            t1 = time.time()
            try:
                mod = __import__(mod_name)
                mod.QUICK = quick
                if quick and hasattr(mod, "SEEDS"):
                    mod.SEEDS = mod.SEEDS[:2]
                mod.main()
                log.append("%s: OK (%.1f min)" % (mod_name, (time.time() - t1) / 60))
            except Exception:
                err = traceback.format_exc()
                print(err)
                log.append("%s: ERROR\n%s" % (mod_name, err))
    print("\n" + "=" * 72 + "\n  Figures\n" + "=" * 72, flush=True)
    try:
        import make_figures
        make_figures.main()
        log.append("make_figures: OK")
    except Exception:
        err = traceback.format_exc()
        print(err)
        log.append("make_figures: ERROR\n%s" % err)
    log.append("total %.1f min | quick=%s" % ((time.time() - t0) / 60, quick))
    with open(os.path.join(HERE, "run_all_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
