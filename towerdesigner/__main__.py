"""Run the app:  python -m towerdesigner           (window)
           or:  python -m towerdesigner --cli     (search in the terminal and export the best design)
"""
import argparse
import os
import sys


def cli(args):
    from .export import export_all
    from .families import FAMILY_NAMES
    from .optimiser import Optimiser
    from .spec import Spec

    from .gluetest import load_glue_factors
    from .woodtest import apply_saved
    spec = Spec(effort=args.effort, height=args.height, max_mass=args.mass, wood=args.wood,
                glue_factors=load_glue_factors())
    if args.wood == "teacher":
        apply_saved(spec)
    if args.families:
        spec.families = args.families.split(",")

    def prog(f, msg, best):
        b = f" | best {FAMILY_NAMES[best.family]} {best.load_kg:.1f} kg" if best else ""
        print(f"\r{f * 100:5.1f}% {msg[:60]:60s}{b}", end="", flush=True)

    opt = Optimiser(spec, progress=prog)
    res = opt.run()
    print()
    print("\n".join(opt.state.log))
    for k, c in enumerate(res[:8], start=1):
        print(f"{k}. {FAMILY_NAMES[c.family]:22s} {c.load_kg:6.1f} kg  {c.mass:5.2f} g  {c.glue}  {c.governing}")
    if res and args.out:
        r = export_all(res[0].family, res[0].params, spec, args.out, progress=lambda m: print("  " + m))
        print("Exported to", os.path.abspath(args.out), *r["notes"], sep="\n")


def main():
    import multiprocessing
    multiprocessing.freeze_support()          # the packaged app runs the optimiser's worker processes too
    if len(sys.argv) > 2 and sys.argv[1] == "--render":   # picture renders run in their own process
        from .render import run
        return run(sys.argv[2:])
    ap = argparse.ArgumentParser(prog="towerdesigner")
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--effort", default="normal", choices=["quick", "normal", "thorough"])
    ap.add_argument("--height", type=float, default=211.0)
    ap.add_argument("--mass", type=float, default=20.0)
    ap.add_argument("--wood", default="teacher")
    ap.add_argument("--families", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("design", nargs="?", default="")
    args = ap.parse_args()
    if args.cli:
        cli(args)
    else:
        from .gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    sys.exit(main())
