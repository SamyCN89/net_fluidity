import os
from pathlib import Path
import subprocess


def test_plot_from_csv(tmp_path: Path):
    # Reuse the synthetic outputs produced by the end-to-end compute test if present,
    # otherwise fabricate a minimal directory with expected CSVs is out of scope here.
    # Instead, we first run the compute test logic by importing it.
    from tests_smoke.test_end_to_end_compute import write_synthetic_context

    ctx = write_synthetic_context(tmp_path)
    env = os.environ.copy()
    env["PATHS_ROOT"] = str(ctx["root"])  # forces fun_paths to use this root
    env["DATASET_NAME"] = "testset"

    # Ensure compute has produced CSVs
    cmd_compute = [
        "python",
        "scripts/compute_speed_bootstrap.py",
        "--tr",
        "500",
        "--subset",
        ctx["subset"],
        "--tau-index",
        "0",
        "--n-boot",
        "20",
        "--q",
        "50",
        "--jobs",
        "1",
        "--parallel-scope",
        "windows",
    ]
    res_c = subprocess.run(cmd_compute, capture_output=True, text=True, env=env, cwd=str(Path.cwd()))
    if res_c.returncode != 0:
        raise AssertionError(f"compute failed: {res_c.returncode}\nSTDOUT:\n{res_c.stdout}\nSTDERR:\n{res_c.stderr}")

    # Run plot script: generate by-window diffs figure
    cmd_plot = [
        "python",
        "scripts/plot_speed_bootstrap.py",
        "--tr",
        "500",
        "--subset",
        ctx["subset"],
        "--plot-diffs-by-win",
        "--plot-diffs-bywin-grid",
        "--bywin-grid-cols",
        "2",
    ]
    res_p = subprocess.run(cmd_plot, capture_output=True, text=True, env=env, cwd=str(Path.cwd()))
    if res_p.returncode != 0:
        raise AssertionError(f"plot failed: {res_p.returncode}\nSTDOUT:\n{res_p.stdout}\nSTDERR:\n{res_p.stderr}")

    # At least one figure should be written under fig/<dataset>/speed/<outdir>
    fig_root = tmp_path / "fig" / "testset" / "speed" / ctx["subset"]
    pngs = list(fig_root.glob("*.png"))
    assert pngs, f"No figures found in {fig_root}"

