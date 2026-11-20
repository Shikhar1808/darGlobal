#!/bin/bash
# EC2 user data (Ubuntu 24.04): Docker, 2 GB swap, the code. API keys are added over SSH afterwards, never here:
# user data is readable by anyone with console access to the instance.
set -eux
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab  # the image build embeds ~900 records; 2 GB RAM alone is tight
apt-get update
apt-get install -y docker.io docker-compose-v2 git
usermod -aG docker ubuntu
git clone https://github.com/Shikhar1808/darGlobal.git /home/ubuntu/darGlobal
chown -R ubuntu:ubuntu /home/ubuntu/darGlobal
