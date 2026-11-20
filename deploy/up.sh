#!/bin/sh
# Start or update the stack on EC2: sh deploy/up.sh   (set DOMAIN=chat.example.com to use your own domain)
set -eu
cd "$(dirname "$0")/.."
[ -s .env ] || { echo "Create .env first: cp .env.example .env, then add your OpenRouter key(s)"; exit 1; }
if [ -z "${DOMAIN:-}" ]; then  # no domain: <public-ip>.sslip.io resolves to this server and gets a real certificate
  t=$(curl -sfX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
  ip=$(curl -sf -H "X-aws-ec2-metadata-token: $t" http://169.254.169.254/latest/meta-data/public-ipv4)
  DOMAIN="$(echo "$ip" | tr . -).sslip.io"
fi
export DOMAIN
git pull --ff-only
docker compose -f docker-compose.yml -f deploy/compose.aws.yml up -d --build
docker image prune -f  # drop the previous app image (1.6 GB each), so updates fit on a small disk
echo "Live at https://$DOMAIN (the first certificate takes up to a minute)"
