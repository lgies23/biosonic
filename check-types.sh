#!/bin/bash

set -e

if ! command -v python3.9 &> /dev/null; then
  echo "Python 3.9 is not installed. Aborting."
  exit 1
fi

if [ ! -d ".venv-py39" ]; then
  echo "Creating Python 3.9 virtual environment..."
  python3.9 -m venv .venv-py39
  source .venv-py39/bin/activate
  pip install -r requirements.txt
else
  source .venv-py39/bin/activate
fi

echo "Python version in use: $(python --version)"
echo "mypy path: $(which mypy)"

mypy src/
