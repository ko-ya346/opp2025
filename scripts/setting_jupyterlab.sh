uv pip install jupyterlab ipykernel

# pyproject.toml に追加
uv add jupyterlab ipykernel

# 仮想環境を jupyter カーネルとして登録
python -m ipykernel install --user --name=opp2025 --display-name "Python (opp2025)"
