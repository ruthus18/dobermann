## Dobermann

*Quantitative trading made simple*

__AT NOW IS IN EARLY WIP__

This is a toolkit for researching and deploying trading strategies in a uniform and efficient way.

**Usage:** check out `examples` folder.

#### Local development

Set up a local environment
```bash
python3 -m venv venv
source venv/bin/activate
poetry install -D
```

Install git-hooks
```
git config core.hooksPath .githooks
```

Set up dependant local serivices
```bash
docker-compose up -d
```

Code linting
```bash
poe pretty
poe lint
```

Launch the Jupyter Lab
```bash
poe lab
```
