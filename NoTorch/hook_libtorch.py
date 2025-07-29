import os
import sys

# Manually parse sys.argv for --TorchLocation
base = None
if "--TorchLocation" in sys.argv:
    idx = sys.argv.index("--TorchLocation")
    if idx + 1 < len(sys.argv):
        base = sys.argv[idx + 1]

# Fallback if not provided or in non-frozen dev
if not base:
    base = getattr(sys, "_MEIPASS", os.getcwd())

# Insert Python path for torch package
torch_py = os.path.join(base, "python")
sys.path.insert(0, torch_py)

# Platform‐specific native library loading
if sys.platform.startswith("win"):
    dll_dir = os.path.join(base, "bin")
    os.add_dll_directory(dll_dir)
else:
    lib_dir = os.path.join(base, "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + existing
