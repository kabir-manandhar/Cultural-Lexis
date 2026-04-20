#!/bin/bash

sessname="jupyter_tunnel"
jupyter_port=13306
nodename="spartan-gpgpu068"

# Clean up any old session
if zellij list-sessions -s 2>/dev/null | grep -qx "$sessname"; then
    zellij kill-session "$sessname"
    zellij delete-session "$sessname" 2>/dev/null || true
    echo "Killed existing zellij session: $sessname"
else
    echo "No existing zellij session named '$sessname' found. Starting a new session."
fi

echo "Starting new zellij session: $sessname"
zellij attach --create --create-background "$sessname"

# Give the server a brief moment to register the session
for _ in {1..20}; do
    zellij list-sessions -s 2>/dev/null | grep -qx "$sessname" && break
    sleep 0.1
done

sleep 5

# Pane 0: Start the SSH tunnel
echo "Starting SSH tunnel to $nodename on port $jupyter_port..."
zellij --session "$sessname" action rename-pane -p terminal_0 "SSH Tunnel"
zellij --session "$sessname" action write-chars -p terminal_0 "ssh -N -L $jupyter_port:localhost:$jupyter_port $USER@$nodename"
zellij --session "$sessname" action send-keys -p terminal_0 "Enter"

# Pane 1: Ssh into the node
echo "Opening SSH session to $nodename..."
pane_id=$(zellij --session "$sessname" action new-pane)
zellij --session "$sessname" action rename-pane -p "$pane_id" "SSH Session"
zellij --session "$sessname" action write-chars -p "$pane_id" "ssh $USER@$nodename"
zellij --session "$sessname" action send-keys -p "$pane_id" "Enter"
# do nested zellij session
sleep 10
nested_sessname="nested_jupyter"
# Kill any existing nested session
zellij --session "$sessname" action write-chars -p "$pane_id" "zellij kill-session $nested_sessname 2>/dev/null || true"
zellij --session "$sessname" action send-keys -p "$pane_id" "Enter"
sleep 2
# Delete any existing nested session
zellij --session "$sessname" action write-chars -p "$pane_id" "zellij delete-session $nested_sessname 2>/dev/null || true"
zellij --session "$sessname" action send-keys -p "$pane_id" "Enter"
sleep 2

zellij --session "$sessname" action write-chars -p "$pane_id" "zellij -s $nested_sessname"
zellij --session "$sessname" action send-keys -p "$pane_id" "Enter"
sleep 5
# call source ~/cd_hua.sh
zellij --session "$sessname" action write-chars -p "$pane_id" "source ~/cd_hua.sh"
zellij --session "$sessname" action send-keys -p "$pane_id" "Enter"
sleep 1
# call jupyter notebook
zellij --session "$sessname" action write-chars -p "$pane_id" "jupyter lab --ServerApp.token='' --ServerApp.password='' --no-browser --port=$jupyter_port"
zellij --session "$sessname" action send-keys -p "$pane_id" "Enter"
