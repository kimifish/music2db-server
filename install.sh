#!/bin/bash

# Exit on error
set -e

# Install the package
pip install -e .

# Create config directory if it doesn't exist
mkdir -p ~/.config/music2db_server

# Copy default config if it doesn't exist
if [ ! -f ~/.config/music2db_server/config.yaml ]; then
    cp config.yaml ~/.config/music2db_server/
fi

# Install systemd service
mkdir -p ~/.config/systemd/user/
cp packaging/music2db-server.service ~/.config/systemd/user/

# Reload systemd daemon
systemctl --user daemon-reload

echo "Installation complete!"
echo "To start the service:"
echo "  systemctl --user enable music2db-server"
echo "  systemctl --user start music2db-server"
echo "To check status:"
echo "  systemctl --user status music2db-server"