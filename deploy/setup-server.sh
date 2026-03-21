#!/usr/bin/env bash
# =============================================================
# One-time VPS setup script
# Run as root on a fresh Ubuntu 22.04 / Debian 12 server
# =============================================================
set -euo pipefail

DEPLOY_USER="trader"
APP_DIR="/home/${DEPLOY_USER}/trading-system"

echo "==> Installing system packages"
apt-get update -qq
apt-get install -y --no-install-recommends \
    curl git ca-certificates gnupg lsb-release ufw fail2ban

# ----------------------------------------------------------
# Docker + Compose plugin
# ----------------------------------------------------------
echo "==> Installing Docker"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable --now docker

# ----------------------------------------------------------
# Unprivileged deploy user
# ----------------------------------------------------------
echo "==> Creating deploy user: ${DEPLOY_USER}"
id -u "${DEPLOY_USER}" &>/dev/null || useradd -m -s /bin/bash "${DEPLOY_USER}"
usermod -aG docker "${DEPLOY_USER}"

# ----------------------------------------------------------
# SSH key for GitHub Actions deploy
# ----------------------------------------------------------
SSH_DIR="/home/${DEPLOY_USER}/.ssh"
mkdir -p "${SSH_DIR}"
chmod 700 "${SSH_DIR}"

echo ""
echo "  Paste the PUBLIC KEY that matches your DEPLOY_SSH_KEY GitHub secret,"
echo "  then press Ctrl-D:"
echo ""
cat >> "${SSH_DIR}/authorized_keys"
chmod 600 "${SSH_DIR}/authorized_keys"
chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "${SSH_DIR}"

# ----------------------------------------------------------
# App directory + env file placeholder
# ----------------------------------------------------------
echo "==> Creating app directory"
sudo -u "${DEPLOY_USER}" git clone \
    https://github.com/YOUR_ORG/trading-system.git "${APP_DIR}" \
    || echo "  (repo already cloned — skipping)"

if [[ ! -f "${APP_DIR}/.env.prod" ]]; then
    cat > "${APP_DIR}/.env.prod" <<'EOF'
# Copy from docs/env-variables.adoc and fill in real values
KITE_API_KEY=
KITE_API_SECRET=
KITE_ACCESS_TOKEN=

POSTGRES_USER=trader
POSTGRES_PASSWORD=CHANGE_ME
POSTGRES_DB=nse_trading
TIMESCALE_URL=postgresql://trader:CHANGE_ME@timescaledb:5432/nse_trading
REDIS_URL=redis://redis:6379/0

TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

GRAFANA_ADMIN_PASSWORD=CHANGE_ME
GRAFANA_ROOT_URL=https://YOUR_DOMAIN/grafana/
MLFLOW_TRACKING_URI=http://mlflow:5000

PAPER_TRADE_MODE=true
EOF
    echo "  Created ${APP_DIR}/.env.prod — fill it in before starting services"
fi

# ----------------------------------------------------------
# Firewall — allow SSH + HTTP/HTTPS only
# ----------------------------------------------------------
echo "==> Configuring UFW firewall"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    comment "SSH"
ufw allow 80/tcp    comment "HTTP (redirect to HTTPS)"
ufw allow 443/tcp   comment "HTTPS"
ufw --force enable

# ----------------------------------------------------------
# Systemd unit — keeps Docker Compose running across reboots
# ----------------------------------------------------------
cat > /etc/systemd/system/trading-system.service <<EOF
[Unit]
Description=NSE Trading System (Docker Compose)
After=docker.service network-online.target
Requires=docker.service

[Service]
User=${DEPLOY_USER}
WorkingDirectory=${APP_DIR}
ExecStart=/usr/bin/docker compose -f docker-compose.yml -f docker-compose.prod.yml up
ExecStop=/usr/bin/docker compose -f docker-compose.yml -f docker-compose.prod.yml down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable trading-system

echo ""
echo "======================================================"
echo "  Server setup complete."
echo ""
echo "  Next steps:"
echo "  1. Edit ${APP_DIR}/.env.prod"
echo "  2. cd ${APP_DIR} && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d"
echo "  3. docker exec -it <timescaledb-container> alembic upgrade head"
echo ""
echo "  Add these secrets to GitHub → Settings → Secrets:"
echo "    DEPLOY_HOST      — server IP or hostname"
echo "    DEPLOY_USER      — ${DEPLOY_USER}"
echo "    DEPLOY_SSH_KEY   — private key matching the public key you just pasted"
echo "    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID — for deploy notifications"
echo "======================================================"
