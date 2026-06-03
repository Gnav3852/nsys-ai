"""Modal scaffold for profiling FastVideo Wan2.1 inference with nsys.

Pass the exact FastVideo command with ``--launch-cmd``. The default working
directory is the cloned FastVideo repository, so commands can reference repo
paths such as ``scripts/inference/inference_wan.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "wan21-nsys-profile"
VOLUME_NAME = "wan21-nsys-profiles"
WORKDIR = "/workspace/FastVideo"
PROFILE_ROOT = "/profiles"
# Reject tiny captures (e.g. nsys segfault before inference finishes).
MIN_PROFILE_REP_BYTES = 1_000_000


NSYS_DEB_URL = (
    "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/"
    "NsightSystems-linux-cli-public-2024.6.1.90-3490548.deb"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "ca-certificates", "build-essential", "ninja-build", "wget")
    .run_commands(
        "wget -q -O /tmp/nsight-systems-cli.deb "
        "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/"
        "NsightSystems-linux-cli-public-2024.6.1.90-3490548.deb",
        "apt-get install -y --no-install-recommends /tmp/nsight-systems-cli.deb",
        "rm -f /tmp/nsight-systems-cli.deb",
        "nsys --version",
    )
    .pip_install("torch", "packaging", "wheel", "setuptools", "ninja", "numpy", "nsys-ai")
)

app = modal.App(APP_NAME)
profiles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _runner(gpu: str):
    return app.function(
        image=image,
        gpu=gpu,
        volumes={PROFILE_ROOT: profiles},
        timeout=60 * 60 * 4,
    )


def _run_shell(cmd: str, *, cwd: str | None = None, check: bool = True) -> int:
    import subprocess

    print(f"==> {cmd}", flush=True)
    result = subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, ["bash", "-lc", cmd])
    return result.returncode


def _commit_profiles() -> None:
    profiles.commit()
    print(f"==> Committed to Modal volume {VOLUME_NAME}", flush=True)


def _validate_profile_artifacts(
    run_dir: Path,
    run_name: str,
    *,
    launch_cmd: str,
    min_rep_bytes: int = MIN_PROFILE_REP_BYTES,
) -> None:
    rep_path = run_dir / f"{run_name}.nsys-rep"
    sqlite_path = run_dir / f"{run_name}.sqlite"

    if not rep_path.is_file():
        raise RuntimeError(f"profile failed: missing {rep_path}")

    rep_size = rep_path.stat().st_size
    if rep_size < min_rep_bytes:
        raise RuntimeError(
            f"partial profile: {rep_path} is {rep_size} bytes "
            f"(minimum {min_rep_bytes}); rerun with fixed nsys settings or a smaller workload first"
        )

    if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
        raise RuntimeError(f"profile failed: missing or empty {sqlite_path}")

    if "fastvideo generate" in launch_cmd:
        videos_dir = run_dir / "videos"
        mp4_files = list(videos_dir.glob("**/*.mp4")) if videos_dir.is_dir() else []
        if not mp4_files:
            raise RuntimeError(
                f"profile incomplete: no .mp4 files under {videos_dir}; "
                "inference likely did not finish"
            )


def _quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def _prepare_repo(repo_url: str, repo_ref: str, setup_cmd: str) -> None:
    from pathlib import Path

    if not repo_url:
        print("==> No --repo-url provided; using container working directory", flush=True)
        return

    repo_path = Path(WORKDIR)
    if not repo_path.exists():
        _run_shell(f"git clone --depth=1 --branch {_quote(repo_ref)} {_quote(repo_url)} {WORKDIR}")
    else:
        print(f"==> Reusing existing checkout at {WORKDIR}", flush=True)

    if setup_cmd:
        _run_shell(setup_cmd, cwd=WORKDIR)


def _profile_or_run(
    *,
    gpu_label: str,
    run_name: str,
    launch_cmd: str,
    repo_url: str,
    repo_ref: str,
    setup_cmd: str,
    profile: bool,
) -> None:
    import json
    import shutil
    import time
    from pathlib import Path

    if not launch_cmd:
        raise ValueError("--launch-cmd is required")

    _prepare_repo(repo_url, repo_ref, setup_cmd)

    run_dir = Path(PROFILE_ROOT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("==> Modal Wan2.1 profiling job")
    print(f"    gpu       : {gpu_label}")
    print(f"    run_name  : {run_name}")
    print(f"    run_dir   : {run_dir}")
    print(f"    repo_url  : {repo_url or '(none)'}")
    print(f"    repo_ref  : {repo_ref}")
    print(f"    profile   : {profile}")
    print(f"    launch_cmd: {launch_cmd}")

    start = time.perf_counter()
    cwd = WORKDIR if Path(WORKDIR).exists() else None

    nsys_profile_rc: int | None = None

    if profile:
        nsys = shutil.which("nsys")
        if not nsys:
            raise RuntimeError(
                "nsys CLI was not found in the Modal image. Use an image with Nsight Systems "
                "CLI installed, or add an image.run_commands(...) step that installs it."
            )

        out_base = run_dir / run_name
        rep_path = out_base.with_suffix(".nsys-rep")
        sqlite_path = out_base.with_suffix(".sqlite")
        nsys_cmd = (
            "nsys profile "
            "--force-overwrite=true "
            "--trace-fork-before=true "
            "--trace=cuda,nvtx,cublas,cudnn "
            "--sample=none "
            "--cpuctxsw=none "
            f"-o {_quote(str(out_base))} "
            f"{launch_cmd}"
        )
        nsys_profile_rc = _run_shell(nsys_cmd, cwd=cwd, check=False)

        if not rep_path.is_file():
            raise RuntimeError(
                f"nsys profile failed (exit {nsys_profile_rc}): no report at {rep_path}"
            )

        rep_size = rep_path.stat().st_size
        if nsys_profile_rc != 0:
            hint = " (SIGSEGV)" if nsys_profile_rc == 139 else ""
            print(
                f"==> WARNING: nsys profile exited {nsys_profile_rc}{hint}; "
                f"report exists ({rep_size} bytes), continuing to export",
                flush=True,
            )

        export_cmd = (
            "nsys export "
            "--force-overwrite=true "
            "--type sqlite "
            f"--output {_quote(str(sqlite_path))} "
            f"{_quote(str(rep_path))}"
        )
        _run_shell(export_cmd, cwd=cwd)
        _commit_profiles()
    else:
        _run_shell(launch_cmd, cwd=cwd)

    end = time.perf_counter()
    timing = {
        "run_name": run_name,
        "gpu": gpu_label,
        "profile": profile,
        "wall_time_s": round(end - start, 3),
        "launch_cmd": launch_cmd,
        "repo_url": repo_url,
        "repo_ref": repo_ref,
        "setup_cmd": setup_cmd,
        "profile_dir": str(run_dir),
    }
    if nsys_profile_rc is not None:
        timing["nsys_profile_rc"] = nsys_profile_rc

    timing_path = run_dir / f"{run_name}_timing.json"
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")

    if profile:
        _validate_profile_artifacts(run_dir, run_name, launch_cmd=launch_cmd)

    print("==> Timing")
    print(json.dumps(timing, indent=2))
    print("==> Artifacts")
    for path in sorted(run_dir.iterdir()):
        print(f"    {path}")

    _commit_profiles()
    print(f"==> Saved to Modal volume {VOLUME_NAME}:/{run_name}/")
    print(f"==> Download with: modal volume get wan21-nsys-profiles /{run_name}/ ...")


@_runner("L40S")
def run_l40s(
    run_name: str,
    launch_cmd: str,
    repo_url: str,
    repo_ref: str,
    setup_cmd: str,
    profile: bool,
) -> None:
    _profile_or_run(
        gpu_label="L40S",
        run_name=run_name,
        launch_cmd=launch_cmd,
        repo_url=repo_url,
        repo_ref=repo_ref,
        setup_cmd=setup_cmd,
        profile=profile,
    )


@_runner("H100")
def run_h100(
    run_name: str,
    launch_cmd: str,
    repo_url: str,
    repo_ref: str,
    setup_cmd: str,
    profile: bool,
) -> None:
    _profile_or_run(
        gpu_label="H100",
        run_name=run_name,
        launch_cmd=launch_cmd,
        repo_url=repo_url,
        repo_ref=repo_ref,
        setup_cmd=setup_cmd,
        profile=profile,
    )


@app.local_entrypoint()
def main(
    gpu: str = "L40S",
    run_name: str = "wan21_l40s_baseline",
    launch_cmd: str = "",
    repo_url: str = "https://github.com/hao-ai-lab/FastVideo.git",
    repo_ref: str = "main",
    setup_cmd: str = "pip install -e .",
    profile: bool = True,
) -> None:
    gpu_key = gpu.upper()
    if gpu_key == "L40S":
        run_l40s.remote(run_name, launch_cmd, repo_url, repo_ref, setup_cmd, profile)
    elif gpu_key == "H100":
        run_h100.remote(run_name, launch_cmd, repo_url, repo_ref, setup_cmd, profile)
    else:
        raise ValueError("gpu must be L40S or H100")
