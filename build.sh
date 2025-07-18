#!/usr/bin/env bash

set -e

# Install Chrome
apt-get update
apt-get install -y wget gnupg ca-certificates
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt install -y ./google-chrome-stable_current_amd64.deb
rm google-chrome-stable_current_amd64.deb

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
