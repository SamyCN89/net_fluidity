#!/usr/bin/env bash
set -euo pipefail

# ===============================================================
# Cluster auto-setup script for SSHFS mounting to central storage
# Usage:  bash mount_cluster.sh funsymania
# ===============================================================

REMOTE_HOST="${1:-funsymania}"
REMOTE_PATH="/mnt/sdc"
LOCAL_MOUNT="$HOME/mnt/${REMOTE_HOST}_sdc"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "[1/6] 🔑 Checking SSH key..."
if [[ ! -f "$SSH_KEY" ]]; then
    echo "[info] Generating SSH key..."
    ssh-keygen -t ed25519 -f "$SSH_KEY" -N ""
fi

echo "[2/6] 📤 Copying SSH key to $REMOTE_HOST..."
ssh-copy-id -i "$SSH_KEY.pub" "samy@$REMOTE_HOST"

echo "[3/6] 📁 Creating local mount point: $LOCAL_MOUNT"
mkdir -p "$LOCAL_MOUNT"

echo "[4/6] 🔗 Testing SSH connectivity..."
ssh -o ConnectTimeout=10 "samy@$REMOTE_HOST" "echo '[remote] Connected OK ✅'"

echo "[5/6] 📦 Mounting remote /mnt/sdc -> $LOCAL_MOUNT"
if mountpoint -q "$LOCAL_MOUNT"; then
    echo "[info] Already mounted."
else
    sshfs "samy@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_MOUNT" || {
        echo "[error] Mount failed. Check SSH or remote path."
        exit 1
    }
fi

echo "[6/6] ⚙️ Adding auto-mount to ~/.bashrc if not already present"
if ! grep -q "$LOCAL_MOUNT" ~/.bashrc; then
    cat >> ~/.bashrc <<EOF

# --- Auto-mount cluster storage ---
if ! mountpoint -q "$LOCAL_MOUNT" && ping -c1 -W1 $REMOTE_HOST >/dev/null 2>&1; then
    sshfs samy@$REMOTE_HOST:$REMOTE_PATH $LOCAL_MOUNT
fi
EOF
    echo "[info] Auto-mount entry added to ~/.bashrc"
else
    echo "[info] Auto-mount already configured."
fi

echo "✅ Done! Verify with: ls $LOCAL_MOUNT/samy/"
