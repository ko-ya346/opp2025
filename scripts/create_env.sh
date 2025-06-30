# install uv
curl -Ls https://astral.sh/uv/install.sh | bash

# initialize project
uv venv
source .venv/bin/activate

# install package
uv pip install -r pyproject.toml
