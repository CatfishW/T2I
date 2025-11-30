#!/bin/bash
# Test script to diagnose reverse SSH tunnel issues
# Run this on the GPU machine (where text2image_server.py runs)

echo "============================================================"
echo "Reverse SSH Tunnel Diagnostic Script"
echo "============================================================"
echo ""

# Check if server is running locally
echo "1. Checking if text2image_server.py is running locally..."
if curl -s --connect-timeout 5 http://localhost:8010/health > /dev/null 2>&1; then
    echo "   ✓ Server is running locally on port 8010"
    response=$(curl -s http://localhost:8010/health)
    echo "   Response: $response"
else
    echo "   ✗ Server is NOT running locally on port 8010"
    echo ""
    echo "   Please start the server first:"
    echo "   python text2image_server.py --host 0.0.0.0 --port 8010"
    exit 1
fi

echo ""

# Check if port is listening
echo "2. Checking if port 8010 is listening..."
if command -v netstat > /dev/null 2>&1; then
    if netstat -tuln 2>/dev/null | grep -q ":8010.*LISTEN"; then
        echo "   ✓ Port 8010 is listening"
        netstat -tuln 2>/dev/null | grep ":8010"
    else
        echo "   ✗ Port 8010 is NOT listening"
    fi
elif command -v ss > /dev/null 2>&1; then
    if ss -tuln 2>/dev/null | grep -q ":8010"; then
        echo "   ✓ Port 8010 is listening"
        ss -tuln 2>/dev/null | grep ":8010"
    else
        echo "   ✗ Port 8010 is NOT listening"
    fi
fi

echo ""

# Check SSH tunnel process
echo "3. Checking for SSH tunnel processes..."
if pgrep -f "ssh.*-R.*8010" > /dev/null; then
    echo "   ✓ Found SSH tunnel process(es)"
    pgrep -af "ssh.*-R.*8010"
elif pgrep -f "plink.*-R.*8010" > /dev/null; then
    echo "   ✓ Found plink tunnel process(es)"
    pgrep -af "plink.*-R.*8010"
else
    echo "   ✗ No SSH tunnel processes found"
    echo "   The tunnel might not be running"
    echo ""
    echo "   Please run setup_reverse_ssh.sh to start the tunnel"
fi

echo ""
echo "============================================================"
echo "Diagnostic Complete"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. On the public server, check SSH config:"
echo "   sudo nano /etc/ssh/sshd_config"
echo "   Make sure 'GatewayPorts yes' is set (or 'GatewayPorts clientspecified')"
echo "   Then restart SSH: sudo systemctl restart sshd"
echo ""
echo "2. On the public server, test the tunnel:"
echo "   curl -v http://localhost:8010/health"
echo "   or"
echo "   curl -v http://127.0.0.1:8010/health"
echo ""

