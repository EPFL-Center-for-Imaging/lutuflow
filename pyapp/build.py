"""
Packages the lutuflow project into a single exectuable with PyApp.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from lutuflow import __version__, __name__

script_directory = Path(__file__).parent

# Set environment variables
os.environ["PYAPP_PROJECT_NAME"] = __name__
os.environ["PYAPP_PROJECT_VERSION"] = __version__
os.environ["PYAPP_PYTHON_VERSION"] = "3.11"
os.environ["PYAPP_EXEC_SCRIPT"] = str(
    (script_directory.parent / "src" / __name__ / "__main__.py").resolve()
)

def build_local_wheel() -> Path:
    """Build a wheel from the local checkout so it can be installed without
    needing this version to be published on PyPI."""
    dist_dir = script_directory.parent / "dist"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", str(script_directory.parent)],
        check=True,
    )
    wheels = list(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel in {dist_dir}, found {wheels}")
    return wheels[0].resolve()


def render_requirements(template_name: str, rendered_name: str, wheel_path: Path) -> str:
    template_path = script_directory / template_name
    rendered_path = script_directory / rendered_name
    content = template_path.read_text()
    content = content.replace(
        "lutuflow[napari]==${PYAPP_PROJECT_VERSION}",
        f"lutuflow[napari] @ {wheel_path.as_uri()}",
    )
    rendered_path.write_text(content)
    return str(rendered_path.resolve())


local_wheel = build_local_wheel()


if os.name == "nt":
    print("Building for Windows.")
    extension = ".exe"  # Windows
    platform = "w64"
    os.environ["PYAPP_PROJECT_DEPENDENCY_FILE"] = render_requirements(
        "requirements.template-windows.txt", "requirements.windows.txt", local_wheel
    )
else:
    print("Building for Linux / MacOS.")
    extension = ""  # Linux and MacOS
    platform = "u64"
    os.environ["PYAPP_PROJECT_DEPENDENCY_FILE"] = render_requirements(
        "requirements.template-linux.txt", "requirements.linux.txt", local_wheel
    )

# Print them
print(f"{os.environ['PYAPP_PROJECT_NAME']=}")
print(f"{os.environ['PYAPP_PROJECT_VERSION']=}")
print(f"{os.environ['PYAPP_PYTHON_VERSION']=}")
print(f"{os.environ['PYAPP_EXEC_SCRIPT']=}")
print(f"{os.environ['PYAPP_PROJECT_DEPENDENCY_FILE']=}")

# Change directory and run cargo build
os.chdir(str(script_directory / "pyapp-latest"))
subprocess.run(["cargo", "build", "--release"], check=True)
os.chdir(str(script_directory))

source_path = str(
    (
        script_directory / "pyapp-latest" / "target" / "release" / f"pyapp{extension}"
    ).resolve()
)
destination_path = str(
    (
        script_directory.parent
        / "release"
        / f"{__name__}_{platform}_{__version__}{extension}"
    ).resolve()
)

if os.path.exists(source_path):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy(source_path, destination_path)
    print(f"Copied {source_path} to {destination_path}")
else:
    print(f"{source_path} does not exist")
