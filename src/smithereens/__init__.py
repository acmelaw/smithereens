"""smithereens — AI coding assistant."""

import os

# Prevent litellm from fetching model pricing from raw.githubusercontent.com
# on every import. Uses the bundled backup JSON instead.
# Override with LITELLM_LOCAL_MODEL_COST_MAP=False to re-enable remote fetching.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

__version__ = "0.1.0"
