#!/bin/sh
# Start or update the stack on EC2: sh deploy/up.sh   (first time with your own domain: DOMAIN=chat.example.com sh deploy/up.sh)
set -eu
cd "$(dirname "$0")/.."
[ -s .env ] || { echo "Create .env first: cp .env.example .env, then add your OpenRouter key(s)"; exit 1; }
if [ -z "${DOMAIN:-}" ]; then DOMAIN=$(sed -n 's/^DOMAIN=//p' .env | tail -n 1); fi  # remembered from the last run
if [ -z "$DOMAIN" ]; then  # no domain: <public-ip>.sslip.io resolves to this server and gets a real certificate
  t=$(curl -sf -m 5 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60") &&
    ip=$(curl -sf -m 5 -H "X-aws-ec2-metadata-token: $t" http://169.254.169.254/latest/meta-data/public-ipv4) ||
    { echo "Can't read the public IP; run: DOMAIN=<ip-with-dashes>.sslip.io sh deploy/up.sh"; exit 1; }
  DOMAIN="$(echo "$ip" | tr . -).sslip.io"
fi
# save in .env, so the next run and plain `docker compose logs caddy` use the same domain and files
sed -i '/^DOMAIN=/d; /^COMPOSE_FILE=/d' .env
[ -z "$(tail -c 1 .env)" ] || echo >> .env
printf 'DOMAIN=%s\nCOMPOSE_FILE=docker-compose.yml:deploy/compose.aws.yml\n' "$DOMAIN" >> .env
git pull --ff-only
docker compose up -d --build
docker image prune -f  # drop the previous app image (1.6 GB each), so updates fit on a small disk
echo "Live at https://$DOMAIN (the first certificate takes up to a minute)"
