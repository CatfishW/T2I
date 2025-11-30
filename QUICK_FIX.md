# Quick Fix: "Empty reply from server"

## Most Likely Cause

The SSH server on the **public server** (vpn.agaii.org) needs to allow reverse port forwarding.

## Quick Fix (2 minutes)

### On the Public Server (vpn.agaii.org):

1. **Edit SSH config:**
   ```bash
   sudo nano /etc/ssh/sshd_config
   ```

2. **Find or add this line:**
   ```
   GatewayPorts yes
   ```
   
   (Or use `GatewayPorts clientspecified` for more security)

3. **Restart SSH service:**
   ```bash
   sudo systemctl restart sshd
   # or
   sudo service ssh restart
   ```

4. **Restart the tunnel on GPU machine:**
   - Stop the existing tunnel (close the minimized window or kill the process)
   - Run `setup_reverse_ssh.bat` again

5. **Test:**
   ```bash
   curl http://localhost:8010/health
   ```

## Verify Server is Running

Before fixing SSH, make sure the server is running on the **GPU machine**:

```bash
# On GPU machine
python text2image_server.py --host 0.0.0.0 --port 8010
```

Then test locally:
```bash
curl http://localhost:8010/health
```

If this works locally but not through the tunnel, it's definitely the SSH GatewayPorts issue.

## Still Not Working?

Run the diagnostic script on the GPU machine:
- Windows: `.\test_tunnel.ps1`
- Linux: `./test_tunnel.sh`

See `TUNNEL_TROUBLESHOOTING.md` for detailed troubleshooting steps.

