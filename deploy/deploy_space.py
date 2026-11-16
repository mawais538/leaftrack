"""Publish the Gradio demo to a Hugging Face Space.

The Space receives ``app.py``, the ``leaftrack`` package (Python files only)
and the files in ``deploy/space/`` (Space card, requirements, system
packages). No weights or dataset files are uploaded: the model is downloaded
from the Hub and example frames from the KOMATSUNA server at runtime.

Usage::

    python deploy/deploy_space.py --repo <user>/leaftrack
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_FILES = ROOT / "deploy" / "space"


def stage(dest: Path) -> list[Path]:
    """Copy everything the Space needs into ``dest`` and return the staged files."""
    shutil.copy2(ROOT / "app.py", dest / "app.py")
    for f in SPACE_FILES.iterdir():
        shutil.copy2(f, dest / f.name)
    for src in (ROOT / "leaftrack").rglob("*.py"):
        target = dest / src.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    shutil.copy2(ROOT / "LICENSE", dest / "LICENSE")
    return sorted(p for p in dest.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--repo", required=True, help="<user>/<space-name>")
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--hardware",
        default="zero-a10g",
        help="ZeroGPU (free for personal accounts, up to 2 Spaces); cpu-basic needs PRO",
    )
    p.add_argument("--dry-run", action="store_true", help="stage and list files only")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        for f in files:
            print("  ", f.relative_to(tmp))
        if args.dry_run:
            return 0
        from huggingface_hub import HfApi

        from leaftrack.hub import use_system_certificates

        use_system_certificates()
        api = HfApi()
        api.create_repo(
            args.repo,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=args.hardware,
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=tmp,
            repo_id=args.repo,
            repo_type="space",
            commit_message="Deploy LeafTrack demo",
            delete_patterns=["leaftrack/**"],  # drop modules removed from the package
        )
    print(f"https://huggingface.co/spaces/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
