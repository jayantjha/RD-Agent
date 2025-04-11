#!/bin/bash

echo "Starting up the container..."
echo "Current working directory: $(pwd)"

echo "source activate kaggle" >> ~/.bashrc
source activate kaggle

echo "Cloning repository from ${RDAGENT_GH_REPO} | branch ${RDAGENT_BRANCH}"

git clone --depth 1 --branch "${RDAGENT_BRANCH}" "${RDAGENT_GH_REPO}" rdagent_repo
pip install -r ./rdagent_repo/requirements.txt
pip install ./rdagent_repo

exec "$@"
