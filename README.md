## Dobermann

*Quantitative trading made simple*

This is a set of tools for researching and deploying trading strategies in a uniform and efficient way.

#### Local development

Set up a local environment
```bash
python3 -m venv venv
source venv/bin/activate
poetry install -D
git config core.hooksPath .githooks
```

Install necessary packages for Jupyter Lab
```
* jupyter labextension install jupyterlab-plotly && jupyter lab build
```

Set up dependant local serivices
```bash
docker-compose up -d
```

Code linting
```bash
poe lint
```

Launch the Jupyter Lab
```bash
poe lab
```
