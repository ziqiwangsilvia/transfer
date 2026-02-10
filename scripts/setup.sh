#!/bin/bash

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing UV"
    curl -LsSf https://astral.sh/uv/0.9.8/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "SETUP: UV exists."
fi

# Install a recent version of python
uv python install 3.13.2
if [ ! -d ".venv" ]; then
    echo "Creating .venv ..."
    uv venv
else
    echo "SETUP: .venv exists"
fi
source .venv/bin/activate

# Dev setup
uv sync
# installs the hooks for YAML and python (ruff, isort)
pre-commit install
# install auto-complete for hydra
eval "$(python cli.py -sc install=bash)"

# HF token access to download gated models, e.g., Llama
if hf auth whoami >/dev/null 2>&1; then
    echo "SETUP: HF authenticated"
else
    echo "Some HF models are gated and require authentication."
    echo "Please generate a token on HF's website then either email it yourself from your personal email or manually enter it below: "
    hf auth login
fi;

# using local config (.git/config) as home directory on pods is not static.
echo "SETUP: configuring ./git/config"
if git config user.email >/dev/null 2>&1; then
    echo "SETUP: git user.email set: $(git config user.email)"
else
    echo -n "git config setup - Enter email: "
    read email
    git config user.email $email
fi;


if git config user.name >/dev/null 2>&1; then
    echo "SETUP: git user.name set: $(git config user.name)"
else
    echo -n "git config setup - Enter name: "
    read name
    git config user.name $name
fi;

# pod only has HTTPS access while apt-get tries via HTTP ...
echo "SETUP: installing apt-get dependencies"
sed -i 's|http://|https://|g' /etc/apt/sources.list && \
sed -i 's|http://|https://|g' /etc/apt/sources.list.d/* 2>/dev/null || true && \
apt-get update;

apt-get install -y tmux htop vim;

# Remind user to complete .env
if [ ! -f ".env" ]; then
    echo "Copying .env.example to .env ..."
    cp .env.example .env
    echo "REMINDER: manually update env variables (HF_TOKEN, HF_HUB)"
else
    echo "SETUP: .env exists so not overriding."
fi
