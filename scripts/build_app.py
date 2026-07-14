#!/usr/bin/env python3
"""Build two self-contained Apple Silicon Preen applications.

Outputs:
  dist/Preen-macos14-arm64.app  (macOS 14.6+, MLX macosx_14_0_arm64)
  dist/Preen-macos26-arm64.app  (macOS 26.2+, MLX macosx_26_0_arm64)

Recipients only need a converted local model. CPython, Preen and all runtime
dependencies are embedded under Contents/Resources/python.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


PBS_RELEASE = "20260623"
PYTHON_VERSION = "3.11.15"
PBS_ARCHIVE = (
    f"cpython-{PYTHON_VERSION}+{PBS_RELEASE}-"
    "aarch64-apple-darwin-install_only_stripped.tar.gz"
)
PBS_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PBS_RELEASE}/cpython-{PYTHON_VERSION}%2B{PBS_RELEASE}-"
    "aarch64-apple-darwin-install_only_stripped.tar.gz"
)
PBS_SHA256 = "2318799eaf104f8a29bc09a93b0851b05dbbcb4ce9a5f045ddea169c0c7ff3a5"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "dist"
DEFAULT_BUILD_ROOT = REPO_ROOT / "build" / "app"


class BuildError(RuntimeError):
    """Expected build or validation failure."""


def log(message: str) -> None:
    print(f"[build_app] {message}", flush=True)


def run(
    args: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [os.fspath(arg) for arg in args]
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if result.returncode != 0:
        detail = ""
        if capture:
            detail = f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        raise BuildError(f"command failed ({result.returncode}): {' '.join(command)}{detail}")
    return result


def output(args: Sequence[str | os.PathLike[str]], *, cwd: Path | None = None) -> str:
    return run(args, cwd=cwd, capture=True).stdout.strip()


def require_commands(names: Iterable[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise BuildError(f"required commands not found: {', '.join(missing)}")


def safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {Path("/").resolve(), REPO_ROOT.resolve(), Path.home().resolve()}
    if resolved in forbidden:
        raise BuildError(f"refusing to remove protected path: {resolved}")
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> str:
    """Hash relative paths, symlink targets and file bytes deterministically."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if path.is_symlink():
            digest.update(b"L")
            target = os.readlink(path).encode("utf-8")
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif path.is_dir():
            digest.update(b"D")
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_gb(byte_count: int) -> str:
    return f"{byte_count / 1e9:.2f}GB"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    for attempt in range(1, 4):
        try:
            run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--output",
                    temporary,
                    url,
                ]
            )
            temporary.replace(destination)
            return
        except BuildError:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt)


def extract_minos(path: Path) -> str:
    text = output(["otool", "-l", path])
    lines = iter(text.splitlines())
    for line in lines:
        if "LC_BUILD_VERSION" not in line:
            continue
        for following in lines:
            match = re.match(r"\s*minos\s+(\S+)", following)
            if match:
                return match.group(1)
            if "cmd " in following:
                break
    raise BuildError(f"LC_BUILD_VERSION/minos not found: {path}")


def write_plist_settings(plist_path: Path, minimum: str) -> None:
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)
    plist["LSMinimumSystemVersion"] = minimum
    plist["LSEnvironment"] = {"PYTHONDWRITEBYTECODE": "1"}
    with plist_path.open("wb") as handle:
        plistlib.dump(plist, handle, fmt=plistlib.FMT_XML, sort_keys=True)


SMOKE_CODE = r"""
import importlib
import importlib.metadata
import importlib.util
import json
import os
import pkgutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm
import statetuner

expected = Path(os.environ["PREEN_VERIFY_PREFIX"]).resolve()
assert Path(sys.prefix).resolve() == expected, (sys.prefix, expected)
assert expected in Path(statetuner.__file__).resolve().parents
assert expected in Path(mlx_lm.__file__).resolve().parents
assert importlib.util.find_spec("torch") is None

modules = sorted(module.name for module in pkgutil.iter_modules(statetuner.__path__))
for name in modules:
    importlib.import_module(f"statetuner.{name}")

package_root = Path(statetuner.__file__).resolve().parent
assert (package_root / "assets" / "rwkv7_hf_template.json").is_file()
assert (package_root / "assets" / "rwkv_world_tokenizer").is_dir()

value = mx.sum(mx.array([1.0, 2.0, 3.0]) ** 2)
mx.eval(value)
assert float(value) == 14.0

print(json.dumps({
    "python": sys.version.split()[0],
    "mlx": importlib.metadata.version("mlx"),
    "mlx_metal": importlib.metadata.version("mlx-metal"),
    "mlx_lm": importlib.metadata.version("mlx-lm"),
    "statetuner_modules": len(modules),
    "metal_sum": float(value),
    "prefix": str(expected),
}, ensure_ascii=False))
"""


class AppBuilder:
    def __init__(self, *, output_dir: Path, build_root: Path, clean: bool) -> None:
        self.output_dir = output_dir.resolve()
        self.build_root = build_root.resolve()
        self.cache_dir = self.build_root / "cache"
        self.work_dir = self.build_root / "work"
        self.log_dir = self.build_root / "logs"
        self.clean = clean
        self.base_dir = self.work_dir / "pbs-base"
        self.base_python = self.base_dir / "python" / "bin" / "python3"
        self.all_requirements = self.work_dir / "requirements-all.txt"
        self.binary_requirements = self.work_dir / "requirements-binary.txt"
        self.common_wheels = self.work_dir / "wheels-common"
        self.mlx_lm_wheel: Path | None = None
        self.preen_wheel: Path | None = None

    def build(self) -> None:
        require_commands(
            [
                "codesign",
                "curl",
                "ditto",
                "file",
                "git",
                "otool",
                "tar",
                "uv",
                "xattr",
                "xcodebuild",
            ]
        )
        if output(["uname", "-m"]) != "arm64":
            raise BuildError("this builder must run on an Apple Silicon Mac")
        if not (REPO_ROOT / "uv.lock").is_file():
            raise BuildError(f"uv.lock not found under {REPO_ROOT}")
        if not (REPO_ROOT / "macos" / "Preen.xcodeproj").is_dir():
            raise BuildError("macos/Preen.xcodeproj not found")

        if self.clean:
            log(f"cleaning {self.build_root}")
            safe_rmtree(self.build_root)
        safe_rmtree(self.work_dir)
        (self.cache_dir / "downloads").mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prepare_base_python()
        self.prepare_requirements_and_common_wheels()
        self.download_platform_wheels("macos14", "macosx_14_0_arm64")
        self.download_platform_wheels("macos26", "macosx_26_0_arm64")
        self.prepare_runtime("macos14")
        self.prepare_runtime("macos26")
        self.build_variant("macos14", os.environ.get("PREEN_MACOS14_TARGET", "14.6"), "14.0")
        self.build_variant("macos26", os.environ.get("PREEN_MACOS26_TARGET", "26.2"), "26.2")
        self.write_manifest()

        if os.environ.get("PREEN_KEEP_BUILD", "0") != "1":
            safe_rmtree(self.work_dir)

        log("done")
        log(f"  {self.output_dir / 'Preen-macos14-arm64.app'}")
        log(f"  {self.output_dir / 'Preen-macos26-arm64.app'}")
        log(f"  {self.output_dir / 'BUILD-MANIFEST.txt'}")

    def prepare_base_python(self) -> None:
        archive = self.cache_dir / "downloads" / PBS_ARCHIVE
        if not archive.is_file():
            log(f"downloading python-build-standalone CPython {PYTHON_VERSION}")
            download(PBS_URL, archive)
        actual = sha256_file(archive)
        if actual != PBS_SHA256:
            raise BuildError(f"PBS SHA-256 mismatch: {actual}")

        self.base_dir.mkdir(parents=True, exist_ok=True)
        run(["tar", "-xzf", archive, "-C", self.base_dir])
        if not os.access(self.base_python, os.X_OK):
            raise BuildError("standalone Python was not extracted correctly")

    def prepare_requirements_and_common_wheels(self) -> None:
        log("exporting locked production dependencies")
        run(
            [
                "uv",
                "export",
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                self.all_requirements,
            ],
            cwd=REPO_ROOT,
        )
        lines = self.all_requirements.read_text(encoding="utf-8").splitlines()
        mlx_lm_spec = next((line for line in lines if line.startswith("mlx-lm @ ")), "")
        if not mlx_lm_spec:
            raise BuildError("mlx-lm direct requirement not found in uv export")
        filtered = [line for line in lines if not line.startswith("mlx-lm @ ")]
        self.binary_requirements.write_text("\n".join(filtered) + "\n", encoding="utf-8")

        self.common_wheels.mkdir(parents=True, exist_ok=True)
        pip_env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        log("building the pinned mlx-lm wheel")
        run(
            [
                self.base_python,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-deps",
                "--wheel-dir",
                self.common_wheels,
                mlx_lm_spec,
            ],
            env=pip_env,
        )

        log("building the current Preen Python wheel")
        run(["uv", "build", "--wheel", "--out-dir", self.common_wheels], cwd=REPO_ROOT)
        self.mlx_lm_wheel = next(self.common_wheels.glob("mlx_lm-*.whl"), None)
        self.preen_wheel = next(self.common_wheels.glob("statetuner-*.whl"), None)
        if self.mlx_lm_wheel is None:
            raise BuildError("mlx-lm wheel was not produced")
        if self.preen_wheel is None:
            raise BuildError("statetuner wheel was not produced")

    def download_platform_wheels(self, label: str, pip_platform: str) -> None:
        wheelhouse = self.work_dir / f"wheelhouse-{label}"
        wheelhouse.mkdir(parents=True, exist_ok=True)
        pip_env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        log(f"downloading locked wheels for {pip_platform}")
        run(
            [
                self.base_python,
                "-m",
                "pip",
                "download",
                "--disable-pip-version-check",
                "--dest",
                wheelhouse,
                "--no-deps",
                "--only-binary=:all:",
                "--platform",
                pip_platform,
                "--python-version",
                "3.11",
                "--implementation",
                "cp",
                "--abi",
                "cp311",
                "--requirement",
                self.binary_requirements,
            ],
            env=pip_env,
        )

    def prepare_runtime(self, label: str) -> None:
        if self.mlx_lm_wheel is None or self.preen_wheel is None:
            raise BuildError("common wheels are not ready")
        runtime = self.work_dir / f"runtime-{label}" / "python"
        wheelhouse = self.work_dir / f"wheelhouse-{label}"
        runtime.parent.mkdir(parents=True, exist_ok=True)
        log(f"assembling Python runtime for {label}")
        run(["ditto", self.base_dir / "python", runtime])
        python = runtime / "bin" / "python3"
        pip_env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--no-compile",
                "--find-links",
                wheelhouse,
                "--requirement",
                self.binary_requirements,
            ],
            env=pip_env,
        )
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--no-compile",
                self.mlx_lm_wheel,
                self.preen_wheel,
            ],
            env=pip_env,
        )
        if (runtime / "lib" / "python3.11" / "site-packages" / "torch").exists():
            raise BuildError(f"torch leaked into {label} runtime")

    def build_variant(self, label: str, deployment_target: str, mlx_minos: str) -> None:
        derived_data = self.work_dir / f"DerivedData-{label}"
        xcode_log = self.log_dir / f"xcode-{label}.log"
        source_app = derived_data / "Build" / "Products" / "Release" / "Preen.app"
        output_app = self.output_dir / f"Preen-{label}-arm64.app"
        runtime = self.work_dir / f"runtime-{label}" / "python"

        log(f"building Swift Release for {label} (minimum macOS {deployment_target})")
        command = [
            "xcodebuild",
            "-project",
            REPO_ROOT / "macos" / "Preen.xcodeproj",
            "-scheme",
            "Preen",
            "-configuration",
            "Release",
            "-destination",
            "generic/platform=macOS",
            "-derivedDataPath",
            derived_data,
            "ARCHS=arm64",
            "ONLY_ACTIVE_ARCH=YES",
            f"MACOSX_DEPLOYMENT_TARGET={deployment_target}",
            "CODE_SIGNING_ALLOWED=NO",
            "CODE_SIGNING_REQUIRED=NO",
            "build",
        ]
        with xcode_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [os.fspath(item) for item in command],
                cwd=REPO_ROOT,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            tail = "\n".join(xcode_log.read_text(encoding="utf-8").splitlines()[-80:])
            raise BuildError(f"xcodebuild failed for {label}; log: {xcode_log}\n{tail}")
        if not source_app.is_dir():
            raise BuildError(f"xcodebuild did not produce {source_app}")

        safe_rmtree(output_app)
        run(["ditto", source_app, output_app])
        embedded_runtime = output_app / "Contents" / "Resources" / "python"
        safe_rmtree(embedded_runtime)
        run(["ditto", runtime, embedded_runtime])
        write_plist_settings(output_app / "Contents" / "Info.plist", deployment_target)

        run(["xattr", "-cr", output_app])
        run(["codesign", "--force", "--deep", "--sign", "-", "--timestamp=none", output_app])
        self.verify_app(label, output_app, deployment_target, mlx_minos)
        log(f"created {output_app} ({format_gb(directory_size(output_app))})")

    def verify_app(self, label: str, app: Path, minimum: str, mlx_minos: str) -> None:
        python = app / "Contents" / "Resources" / "python" / "bin" / "python3"
        executable = app / "Contents" / "MacOS" / "Preen"
        plist_path = app / "Contents" / "Info.plist"
        libmlx = (
            app
            / "Contents"
            / "Resources"
            / "python"
            / "lib"
            / "python3.11"
            / "site-packages"
            / "mlx"
            / "lib"
            / "libmlx.dylib"
        )
        for path in (python, executable, plist_path, libmlx):
            if not path.exists():
                raise BuildError(f"missing from {label} app: {path}")
        if "arm64" not in output(["file", python]):
            raise BuildError(f"{label} Python is not arm64")
        if "arm64" not in output(["file", executable]):
            raise BuildError(f"{label} Swift executable is not arm64")
        with plist_path.open("rb") as handle:
            plist = plistlib.load(handle)
        if plist.get("LSMinimumSystemVersion") != minimum:
            raise BuildError(f"{label} plist minimum is not {minimum}")
        executable_minos = extract_minos(executable)
        if executable_minos != minimum:
            raise BuildError(
                f"{label} Swift executable minos is {executable_minos}, expected {minimum}"
            )
        actual_mlx_minos = extract_minos(libmlx)
        if actual_mlx_minos != mlx_minos:
            raise BuildError(
                f"{label} libmlx minos is {actual_mlx_minos}, expected {mlx_minos}"
            )

        verify_root = self.work_dir / f"verify-{label}"
        home = verify_root / "home"
        temporary = verify_root / "tmp"
        home.mkdir(parents=True, exist_ok=True)
        temporary.mkdir(parents=True, exist_ok=True)
        environment = {
            "HOME": os.fspath(home),
            "TMPDIR": os.fspath(temporary),
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PREEN_VERIFY_PREFIX": os.fspath(
                app / "Contents" / "Resources" / "python"
            ),
        }
        log(f"running isolated Python/Metal smoke test for {label}")
        run([python, "-c", SMOKE_CODE], env=environment)
        run(["codesign", "--verify", "--deep", "--strict", app])

    def write_manifest(self) -> None:
        macos14 = self.output_dir / "Preen-macos14-arm64.app"
        macos26 = self.output_dir / "Preen-macos26-arm64.app"
        commit = output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        dirty = bool(output(["git", "status", "--short"], cwd=REPO_ROOT))
        manifest = self.output_dir / "BUILD-MANIFEST.txt"
        manifest.write_text(
            "\n".join(
                [
                    "Preen self-contained app build",
                    f"git_commit={commit}",
                    f"git_dirty={str(dirty).lower()}",
                    f"python={PYTHON_VERSION}",
                    f"python_build_standalone_release={PBS_RELEASE}",
                    f"python_build_standalone_sha256={PBS_SHA256}",
                    f"macos14_minimum={os.environ.get('PREEN_MACOS14_TARGET', '14.6')}",
                    f"macos26_minimum={os.environ.get('PREEN_MACOS26_TARGET', '26.2')}",
                    f"macos14_app_tree_sha256={tree_digest(macos14)}",
                    f"macos26_app_tree_sha256={tree_digest(macos26)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build macOS 14 and macOS 26 self-contained Preen.app bundles."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove cached PBS downloads and rebuild everything",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("PREEN_APP_OUTPUT_DIR", DEFAULT_OUTPUT)),
        help="output directory (default: ./dist)",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path(os.environ.get("PREEN_APP_BUILD_ROOT", DEFAULT_BUILD_ROOT)),
        help="cache and temporary build directory (default: ./build/app)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        AppBuilder(
            output_dir=args.output_dir,
            build_root=args.build_root,
            clean=args.clean,
        ).build()
    except (BuildError, OSError) as error:
        print(f"[build_app] error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
