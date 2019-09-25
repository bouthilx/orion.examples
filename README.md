# Basic MNIST Example

You can find here examples for the commandline API and python APIs (worker, service and framework
APIs).

## Installation

```bash
pip install -r requirements.txt
```

## Commandline API

```bash
orion hunt -n cmdline-api-mnist-example \
    ./commandline_api.py --lr~'loguniform(1e-5,1.0)' --momentum~'uniform(0,1)'
```

## Python API

```bash
python worker_api.py [OPTIONS]
python service_api.py [OPTIONS]
python framework_api.py [OPTIONS]
```
