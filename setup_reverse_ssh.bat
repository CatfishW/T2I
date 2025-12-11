@echo off
chcp 65001 >nul 2>&1

REM Text2Image Server Reverse SSH Tunnel Maintainer (Windows version)
REM This script establishes and maintains a reverse SSH tunnel from your Text2Image server
REM (Machine A with private IP) to your web server (Machine B with public IP)
REM Enables remote text-to-image inference on port 8010

SETLOCAL EnableDelayedExpansion

REM ============================================================================
REM CONFIGURATION - Edit these values for your setup
REM ============================================================================

SET REMOTE_USER=lobin
SET REMOTE_HOST=vpn.agaii.org
SET REMOTE_PORT=8010
SET LOCAL_PORT=8010
SET KEEPALIVE_INTERVAL=30
SET KEEPALIVE_MAX=3
SET RECONNECT_DELAY=5

REM ============================================================================

echo ====================================================================
echo       Text2Image Server Reverse SSH Tunnel Maintainer
echo ====================================================================
echo.

REM Check if configuration is default
if "%REMOTE_HOST%"=="your-server-ip" (
    echo [ERROR] Please edit this script and update the CONFIGURATION section!
    echo [INFO] Set REMOTE_USER and REMOTE_HOST to match your web server
    pause
    exit /b 1
)

REM Check if SSH is available
where ssh >nul 2>nul
if errorlevel 1 (
    echo [ERROR] SSH is not available. Please install OpenSSH or use Git Bash.
    echo [INFO] Windows 10/11: Settings ^> Apps ^> Optional Features ^> Add OpenSSH Client
    pause
    exit /b 1
)

REM Check if remote hostname can be resolved
echo [INFO] Checking DNS resolution for %REMOTE_HOST%...
ping -n 1 %REMOTE_HOST% >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot resolve hostname: %REMOTE_HOST%
    echo [INFO] DNS resolution failed. Possible issues:
    echo   - Network connectivity problem
    echo   - Incorrect hostname: %REMOTE_HOST%
    echo   - DNS server not responding
    echo   - VPN or firewall blocking DNS queries
    echo.
    echo [INFO] Attempting DNS lookup...
    nslookup %REMOTE_HOST% 2>nul
    echo.
    set /p CONTINUE="Continue anyway? (y/N): "
    if /i not "!CONTINUE!"=="y" exit /b 1
) else (
    echo [SUCCESS] Hostname %REMOTE_HOST% resolved successfully
)

REM Test basic SSH connectivity
echo [INFO] Testing SSH connectivity to %REMOTE_USER%@%REMOTE_HOST%...
ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no "%REMOTE_USER%@%REMOTE_HOST%" exit >nul 2>nul
if errorlevel 1 (
    echo [WARNING] SSH connectivity test failed or requires authentication
    echo [INFO] This is normal if key-based auth is not set up
    echo [INFO] SSH will prompt for password when connecting
) else (
    echo [SUCCESS] SSH connectivity confirmed
)

REM Check if local Text2Image server is running
echo [INFO] Checking local Text2Image server on port %LOCAL_PORT%...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://127.0.0.1:%LOCAL_PORT%/health' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; exit 0 } catch { exit 1 }" >nul 2>nul
if errorlevel 1 (
    echo [WARNING] Local Text2Image server not responding on port %LOCAL_PORT%
    echo [WARNING] Make sure text2image_server.py is running before starting tunnel
    set /p CONTINUE="Continue anyway? (y/N): "
    if /i not "!CONTINUE!"=="y" exit /b 1
) else (
    echo [SUCCESS] Local Text2Image server is running on port %LOCAL_PORT%
)

echo.
echo [INFO] Configuration:
echo   Remote Server: %REMOTE_USER%@%REMOTE_HOST%
echo   Remote Port:   %REMOTE_PORT% (127.0.0.1 on Machine B)
echo   Local Port:    %LOCAL_PORT% (Text2Image server)
echo   Keepalive:     %KEEPALIVE_INTERVAL%s interval, %KEEPALIVE_MAX% max failures
echo   Reconnect:     %RECONNECT_DELAY%s delay
echo.

echo [SUCCESS] Tunnel starting...
echo.

echo [INFO] What's happening:
echo   - REVERSE TUNNEL: Makes your LOCAL Text2Image server accessible on REMOTE server
echo   - Your local server (127.0.0.1:%LOCAL_PORT% on THIS machine) is forwarded to:
echo   - Remote server can access it at http://127.0.0.1:%REMOTE_PORT% (ONLY on remote server)
echo   - This connection is encrypted and secure
echo.

echo [IMPORTANT] Reverse tunnel direction:
echo   THIS MACHINE (Local)          REMOTE SERVER
echo   ------------------            ------------------
echo   Text2Image :%LOCAL_PORT%  --^>  Accessible at 127.0.0.1:%REMOTE_PORT%
echo.

echo [INFO] To verify on REMOTE SERVER (Machine B):
echo   1. SSH to remote: ssh %REMOTE_USER%@%REMOTE_HOST%
echo   2. On remote server, run: curl http://127.0.0.1:%REMOTE_PORT%/health
echo.

echo [NOTE] You CANNOT access remote services through this reverse tunnel!
echo        This tunnel only exposes YOUR local service to the remote server.
echo        To access remote services locally, you need a FORWARD tunnel (^>L^).
echo.

echo [WARNING] Keep this window open! Closing it will stop the tunnel.
echo.

:TUNNEL_LOOP

SET /a ATTEMPT+=1

echo [INFO] Starting tunnel (attempt #%ATTEMPT%) at %TIME%...
echo [INFO] Forwarding: %REMOTE_HOST%:%REMOTE_PORT% -^> 127.0.0.1:%LOCAL_PORT%

REM Build connection string
SET SSH_CONNECT=%REMOTE_USER%@%REMOTE_HOST%
echo [DEBUG] Connection string: %SSH_CONNECT%
echo.

REM Start SSH tunnel (single line to avoid encoding issues)
REM Using -F NUL to bypass SSH config file which might have encoding issues
ssh -F NUL -R %REMOTE_PORT%:127.0.0.1:%LOCAL_PORT% -o ServerAliveInterval=%KEEPALIVE_INTERVAL% -o ServerAliveCountMax=%KEEPALIVE_MAX% -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no -o LogLevel=ERROR -N "%SSH_CONNECT%" 2>&1

SET EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==0 (
    echo [INFO] Tunnel stopped gracefully
    goto :END
)

echo [WARNING] Tunnel disconnected at %TIME% (exit code: %EXIT_CODE%)

REM Check if it's a DNS resolution error
if %EXIT_CODE%==255 (
    echo [ERROR] Connection failed. Possible causes:
    echo   - DNS resolution failed for %REMOTE_HOST%
    echo   - Network connectivity issue
    echo   - SSH server not reachable on %REMOTE_HOST%
    echo   - Incorrect hostname or IP address
    echo.
    echo [INFO] Testing hostname resolution again...
    ping -n 1 %REMOTE_HOST% >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Hostname %REMOTE_HOST% cannot be resolved!
        echo [INFO] Please check:
        echo   1. Internet connectivity
        echo   2. VPN connection (if required)
        echo   3. Firewall settings
        echo   4. Hostname spelling: %REMOTE_HOST%
        echo.
    ) else (
        echo [INFO] Hostname resolves, but SSH connection failed.
        echo [INFO] Check SSH server is running and credentials are correct.
        echo.
    )
)

echo [INFO] Reconnecting in %RECONNECT_DELAY% seconds...
timeout /t %RECONNECT_DELAY% /nobreak >nul

goto :TUNNEL_LOOP

:END
echo [INFO] Tunnel stopped
pause
