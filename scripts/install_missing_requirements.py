"""Install only absent distributions without changing packages in the active environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import shlex
import subprocess
import sys
from pathlib import Path


def load_requirements(path: Path):
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError as exc:
        raise ValueError(
            "This script requires packaging in the active environment. "
            "Install it explicitly with: python -m pip install packaging"
        ) from exc

    requirements = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise ValueError(f"{path}:{number}: expected a named PEP 508 requirement, not pip options") from exc
        if requirement.url is not None:
            raise ValueError(f"{path}:{number}: direct URL requirements are not supported")
        requirements.append(requirement)
    return requirements


def installed_packages():
    from packaging.utils import canonicalize_name

    return {
        canonicalize_name(name): distribution.version
        for distribution in importlib.metadata.distributions()
        if (name := distribution.metadata.get("Name"))
    }


def plan_install(requirements, installed):
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion

    missing = []
    for requirement in requirements:
        if requirement.marker and not requirement.marker.evaluate(environment={"extra": ""}):
            print(f"SKIP marker: {requirement}")
            continue
        name = canonicalize_name(requirement.name)
        if name not in installed:
            missing.append(str(requirement))
            print(f"MISSING: {requirement}")
            continue
        version = installed[name]
        try:
            compatible = requirement.specifier.contains(version, prereleases=True)
        except InvalidVersion:
            compatible = False
        suffix = "" if compatible else f" [version differs from {requirement.specifier}; kept unchanged]"
        print(f"SKIP installed: {requirement.name}=={version}{suffix}")
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=Path(__file__).resolve().parents[1] / "requirements.txt")
    parser.add_argument("--index-url", help="Optional package index URL; otherwise use pip's configured index")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running pip")
    args = parser.parse_args(argv)
    print(f"Python: {sys.executable}")
    try:
        # Validate the entire input before invoking pip, including inactive requirements.
        requirements = load_requirements(args.requirements)
        missing = plan_install(requirements, installed_packages())
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    command = [sys.executable, "-m", "pip", "install", "--no-deps"]
    if args.index_url:
        command.extend(["--index-url", args.index_url])
    command.extend(missing)
    print(f"Missing requirements: {len(missing)}")
    if missing:
        print(f"Install command: {shlex.join(command)}")
    if args.dry_run:
        print("Dry run: no packages installed or changed; dependency checks were not run.")
        return 0
    if missing:
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    print("Checking declared dependencies; this does not validate project compatibility or GPU execution.", flush=True)
    result = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
    if result.returncode:
        print("Dependency conflicts remain. Existing packages were not automatically replaced.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
