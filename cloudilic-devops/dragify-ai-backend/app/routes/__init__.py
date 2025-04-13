import importlib
import os
import glob
from fastapi import APIRouter

router = APIRouter()

# Get the absolute path of the routes directory
routes_dir = os.path.dirname(__file__)

# Find all Python files recursively in routes (excluding __init__.py)
route_files = [
    os.path.relpath(f, start=routes_dir)[:-3]  # Remove `.py` extension
    for f in glob.glob(os.path.join(routes_dir, "**", "*.py"), recursive=True)
    if not f.endswith("__init__.py")
]

# Convert file paths to module paths (fixing Windows paths)
for module_name in route_files:
    module_name = module_name.replace(os.sep, ".")  # Convert to dotted module path
    module_name = f"app.routes.{module_name}"  # Ensure correct import path

    try:
        module = importlib.import_module(module_name)

        if hasattr(module, "router"):
            prefix = getattr(module, "route_metadata", {}).get("prefix", "")
            tags = getattr(module, "route_metadata", {}).get("tags", [])
            router.include_router(module.router, prefix=prefix, tags=tags)
    
    except ModuleNotFoundError as e:
        print(f"⚠️ Warning: Could not import {module_name}. Error: {e}")
