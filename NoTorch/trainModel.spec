# -*- mode: python ; coding: utf-8 -*-
import os, sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
project_dir =os.getcwd()

# 1) Path setup
pathex = [project_dir]
site_packages = next(p for p in sys.path if "site-packages" in p)

# 2) Exclude torch so it’s always external
excludes = [
    "torch", "torch.*",
    "torchvision", "torchvision.*",
    "torchaudio", "torchaudio.*",
    "torchtext", "torchtext.*",
]

# 3) Runtime‐hook to inject your external libtorch folder
runtime_hooks = [
    os.path.join(project_dir, "hook_libtorch.py"),
]

# 4) Hidden imports (if you need any; PyInstaller often picks these up automatically)
hiddenimports = collect_submodules("cv2")  # example for OpenCV

# 5) Data files (non-binary) you might need
datas = []

# 6) Scan site-packages for EVERY .dll/.so/.pyd except torch’s
binaries = []
for root, dirs, files in os.walk(site_packages):
    # skip Torch entirely
    if "torch" in root.lower():
        continue

    for fn in files:
        if fn.lower().endswith((".pyd", ".dll", ".so", ".dylib")):
            src = os.path.join(root, fn)
            # preserve the relative path under site-packages
            dest = os.path.relpath(root, site_packages)
            binaries.append((src, dest))


a = Analysis(
    ["trainModel.py"],
    pathex=pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=excludes,
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
	a.binaries,
    [],                  # Already handed in all binaries above
    a.zipfiles,
	a.datas,
    name="trainModel",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
	onefile=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)