# Test script to diagnose reverse SSH tunnel issues
# Run this on the GPU machine (where text2image_server.py runs)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Reverse SSH Tunnel Diagnostic Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if server is running locally
Write-Host "1. Checking if text2image_server.py is running locally..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8010/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    Write-Host "   ✓ Server is running locally on port 8010" -ForegroundColor Green
    Write-Host "   Response: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "   Content: $($response.Content)" -ForegroundColor Gray
} catch {
    Write-Host "   ✗ Server is NOT running locally on port 8010" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "   Please start the server first:" -ForegroundColor Yellow
    Write-Host "   python text2image_server.py --host 0.0.0.0 --port 8010" -ForegroundColor White
    exit 1
}

Write-Host ""

# Check if port is listening
Write-Host "2. Checking if port 8010 is listening..." -ForegroundColor Yellow
$listening = Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    Write-Host "   ✓ Port 8010 is listening" -ForegroundColor Green
    Write-Host "   Local Address: $($listening.LocalAddress)" -ForegroundColor Gray
} else {
    Write-Host "   ✗ Port 8010 is NOT listening" -ForegroundColor Red
    Write-Host "   The server might not be bound correctly" -ForegroundColor Red
}

Write-Host ""

# Check SSH tunnel process
Write-Host "3. Checking for SSH tunnel processes..." -ForegroundColor Yellow
$plinkProcesses = Get-Process -Name "plink" -ErrorAction SilentlyContinue
$sshProcesses = Get-Process -Name "ssh" -ErrorAction SilentlyContinue

if ($plinkProcesses) {
    Write-Host "   ✓ Found $($plinkProcesses.Count) plink.exe process(es)" -ForegroundColor Green
    foreach ($proc in $plinkProcesses) {
        Write-Host "     PID: $($proc.Id), Started: $($proc.StartTime)" -ForegroundColor Gray
    }
} elseif ($sshProcesses) {
    Write-Host "   ✓ Found $($sshProcesses.Count) ssh.exe process(es)" -ForegroundColor Green
    foreach ($proc in $sshProcesses) {
        Write-Host "     PID: $($proc.Id), Started: $($proc.StartTime)" -ForegroundColor Gray
    }
} else {
    Write-Host "   ✗ No SSH tunnel processes found" -ForegroundColor Red
    Write-Host "   The tunnel might not be running" -ForegroundColor Red
    Write-Host ""
    Write-Host "   Please run setup_reverse_ssh.bat to start the tunnel" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Diagnostic Complete" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. On the public server, check SSH config:" -ForegroundColor White
Write-Host "   sudo nano /etc/ssh/sshd_config" -ForegroundColor Gray
Write-Host "   Make sure 'GatewayPorts yes' is set (or 'GatewayPorts clientspecified')" -ForegroundColor Gray
Write-Host "   Then restart SSH: sudo systemctl restart sshd" -ForegroundColor Gray
Write-Host ""
Write-Host "2. On the public server, test the tunnel:" -ForegroundColor White
Write-Host "   curl -v http://localhost:8010/health" -ForegroundColor Gray
Write-Host "   or" -ForegroundColor Gray
Write-Host "   curl -v http://127.0.0.1:8010/health" -ForegroundColor Gray
Write-Host ""

