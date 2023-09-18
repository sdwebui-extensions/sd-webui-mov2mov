import os
import launch

# Install requirements if not installed
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            launch.run_pip(f"install {lib}", f"requirement for mov2mov: {lib}")
