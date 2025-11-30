# Reverse SSH Tunnel Troubleshooting Guide

## Problem: "Empty reply from server"

If you're getting `curl: (52) Empty reply from server` when trying to access the tunneled service, here are the steps to diagnose and fix:

### Step 1: Verify Server is Running on GPU Machine

On the **GPU machine** (where `text2image_server.py` runs), check if the server is running:

```bash
# Windows PowerShell
curl http://localhost:8010/health

# Or use the diagnostic script
.\test_tunnel.ps1
```

**If the server is not running:**
```bash
python text2image_server.py --host 0.0.0.0 --port 8010
```

**Important:** The server must bind to `0.0.0.0`, not `127.0.0.1`, to accept connections through the tunnel.

### Step 2: Check SSH Tunnel Process

On the **GPU machine**, verify the tunnel is running:

**Windows:**
```powershell
Get-Process | Where-Object {$_.ProcessName -like "*plink*" -or $_.ProcessName -like "*ssh*"}
```

**Linux:**
```bash
ps aux | grep "ssh.*-R.*8010"
```

If no tunnel process is found, restart the tunnel:
- Windows: Run `setup_reverse_ssh.bat`
- Linux: Run `setup_reverse_ssh.sh`

### Step 3: Configure SSH Server (Public Server)

The most common issue is that the **public server's SSH daemon** needs to allow reverse port forwarding.

On the **public server** (vpn.agaii.org), edit the SSH config:

```bash
sudo nano /etc/ssh/sshd_config
```

Find or add this line:
```
GatewayPorts yes
```

Or for more security (only allow specific users):
```
GatewayPorts clientspecified
```

**Then restart the SSH service:**
```bash
sudo systemctl restart sshd
# or
sudo service ssh restart
```

### Step 4: Test the Tunnel

On the **public server**, test the tunnel:

```bash
# Test with verbose output
curl -v http://localhost:8010/health

# Or test with 127.0.0.1
curl -v http://127.0.0.1:8010/health
```

**Expected output:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  ...
}
```

### Step 5: Check SSH Logs

If still not working, check SSH logs on both machines:

**On public server:**
```bash
sudo tail -f /var/log/auth.log
# or
sudo journalctl -u ssh -f
```

**On GPU machine:**
Check the tunnel process output (if running in foreground) or check Windows Event Viewer.

### Common Issues and Solutions

#### Issue 1: "Connection refused"
- **Cause:** Server not running on GPU machine
- **Solution:** Start `text2image_server.py` on GPU machine

#### Issue 2: "Empty reply from server"
- **Cause:** SSH server doesn't allow reverse forwarding, or server bound to wrong interface
- **Solution:** 
  1. Enable `GatewayPorts yes` in `/etc/ssh/sshd_config` on public server
  2. Restart SSH service
  3. Ensure server binds to `0.0.0.0`, not `127.0.0.1`

#### Issue 3: "Port already in use"
- **Cause:** Another process is using port 8010 on public server
- **Solution:** 
  ```bash
  # Find what's using the port
  sudo lsof -i :8010
  # or
  sudo netstat -tulpn | grep 8010
  
  # Kill the process or use a different port
  ```

#### Issue 4: Tunnel disconnects frequently
- **Cause:** Network instability or SSH keepalive not configured
- **Solution:** The setup script already includes keepalive options:
  - `ServerAliveInterval=60`
  - `ServerAliveCountMax=3`
  
  If still disconnecting, increase these values or check network stability.

### Alternative: Use SSH Config File

Instead of command-line options, you can use an SSH config file for more reliable connections:

**On GPU machine, create/edit `~/.ssh/config` (or `C:\Users\YourName\.ssh\config` on Windows):**

```
Host vpn-tunnel
    HostName vpn.agaii.org
    User lobin
    Port 22
    ServerAliveInterval 60
    ServerAliveCountMax 3
    RemoteForward 8010 localhost:8010
```

Then connect with:
```bash
ssh -N vpn-tunnel
```

### Testing the Full Stack

Once the tunnel is working, test the full flow:

1. **On public server:**
   ```bash
   curl http://localhost:8010/health
   ```

2. **From frontend (if configured):**
   ```bash
   curl http://localhost:3000/api/health
   ```

### Still Not Working?

1. Run the diagnostic script on the GPU machine:
   - Windows: `.\test_tunnel.ps1`
   - Linux: `./test_tunnel.sh`

2. Check firewall rules on both machines

3. Verify network connectivity between machines:
   ```bash
   # From GPU machine
   ping vpn.agaii.org
   ```

4. Check if the port is actually forwarded:
   ```bash
   # On public server
   sudo netstat -tulpn | grep 8010
   # Should show something like: 127.0.0.1:8010 or 0.0.0.0:8010
   ```

