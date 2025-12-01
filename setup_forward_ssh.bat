@echo off
chcp 65001 >nul 2>&1

REM Text2Image Server Forward SSH Tunnel (Windows version)

REM This script creates a FORWARD SSH tunnel to access a REMOTE service locally

REM Use this if you want to access a service on the remote server from your local machine

SETLOCAL EnableDelayedExpansion

REM ============================================================================

REM CONFIGURATION - Edit these values for your setup

REM ============================================================================

SET REMOTE_USER=lobin

SET REMOTE_HOST=vpn.agaii.org

REM Remote service port on the remote server (e.g., Text2Image server on remote)

SET REMOTE_SERVICE_PORT=8010

REM Local port where you want to access the remote service

SET LOCAL_PORT=8010

SET KEEPALIVE_INTERVAL=30

SET KEEPALIVE_MAX=3

SET RECONNECT_DELAY=5

REM ============================================================================

echo ╔════════════════════════════════════════════════════════════╗

echo ║      Text2Image Forward SSH Tunnel Maintainer              ║

echo ╚════════════════════════════════════════════════════════════╝

echo.

echo [INFO] FORWARD TUNNEL: Makes REMOTE service accessible LOCALLY

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

    echo   • Network connectivity problem

    echo   • Incorrect hostname: %REMOTE_HOST%

    echo   • DNS server not responding

    echo   • VPN or firewall blocking DNS queries

    echo.

    echo [INFO] Attempting DNS lookup...

    nslookup %REMOTE_HOST% 2>nul

    echo.

    set /p CONTINUE="Continue anyway? (y/N): "

    if /i not "!CONTINUE!"=="y" exit /b 1

) else (

    echo [SUCCESS] Hostname %REMOTE_HOST% resolved successfully

)

echo.

echo [INFO] Configuration:

echo   Remote Server: %REMOTE_USER%@%REMOTE_HOST%

echo   Remote Service Port: %REMOTE_SERVICE_PORT% (on remote server)

echo   Local Access Port:   %LOCAL_PORT% (on this machine)

echo   Keepalive:           %KEEPALIVE_INTERVAL%s interval, %KEEPALIVE_MAX% max failures

echo   Reconnect:           %RECONNECT_DELAY%s delay

echo.

echo [SUCCESS] Tunnel starting...

echo.

echo [INFO] What's happening:

echo   • FORWARD TUNNEL: Makes REMOTE service accessible on YOUR local machine

echo   • Remote server service (127.0.0.1:%REMOTE_SERVICE_PORT% on remote) is forwarded to:

echo   • Your local machine can access it at http://127.0.0.1:%LOCAL_PORT%

echo   • This connection is encrypted and secure

echo.

echo [INFO] Forward tunnel direction:

echo   THIS MACHINE (Local)          REMOTE SERVER

echo   ──────────────────            ──────────────

echo   Accessible at :%LOCAL_PORT% <──  Text2Image :%REMOTE_SERVICE_PORT%

echo.

echo [INFO] To test from THIS machine after tunnel starts:

echo   curl http://127.0.0.1:%LOCAL_PORT%/health

echo.

echo [INFO] To use remote Text2Image service:

echo   curl http://127.0.0.1:%LOCAL_PORT%/generate ^

echo     -H "Content-Type: application/json" ^

echo     -d "{\"prompt\": \"your prompt here\", \"model\": \"your-model\"}"

echo.

echo [WARNING] Keep this window open! Closing it will stop the tunnel.

echo.

:TUNNEL_LOOP

SET /a ATTEMPT+=1

echo [INFO] Starting tunnel (attempt #%ATTEMPT%) at %TIME%...

echo [INFO] Forwarding: 127.0.0.1:%LOCAL_PORT% (local) <^> %REMOTE_HOST%:127.0.0.1:%REMOTE_SERVICE_PORT% (remote)

echo.

REM Build connection string

SET SSH_CONNECT=%REMOTE_USER%@%REMOTE_HOST%

echo [DEBUG] Connection string: %SSH_CONNECT%

echo.

REM Start SSH forward tunnel (single line to avoid encoding issues)

REM Using -F NUL to bypass SSH config file which might have encoding issues

REM -L creates FORWARD tunnel: local_port:remote_host:remote_port

ssh -F NUL -L %LOCAL_PORT%:127.0.0.1:%REMOTE_SERVICE_PORT% -o ServerAliveInterval=%KEEPALIVE_INTERVAL% -o ServerAliveCountMax=%KEEPALIVE_MAX% -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no -o LogLevel=ERROR -N "%SSH_CONNECT%" 2>&1

SET EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==0 (

    echo [INFO] Tunnel stopped gracefully

    goto :END

)

echo [WARNING] Tunnel disconnected at %TIME% (exit code: %EXIT_CODE%)

REM Check if it's a DNS resolution error

if %EXIT_CODE%==255 (

    echo [ERROR] Connection failed. Possible causes:

    echo   • DNS resolution failed for %REMOTE_HOST%

    echo   • Network connectivity issue

    echo   • SSH server not reachable on %REMOTE_HOST%

    echo   • Incorrect hostname or IP address

    echo   • Remote service not running on port %REMOTE_SERVICE_PORT%

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

        echo [INFO] Also verify remote service is running on port %REMOTE_SERVICE_PORT%

        echo.

    )

)

echo [INFO] Reconnecting in %RECONNECT_DELAY% seconds...

timeout /t %RECONNECT_DELAY% /nobreak >nul

goto :TUNNEL_LOOP

:END

echo [INFO] Tunnel stopped

pause


