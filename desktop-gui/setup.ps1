$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Native command failures do not automatically stop Windows PowerShell.
function Invoke-Kubectl {
    & kubectl @args
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl failed with exit code $LASTEXITCODE"
    }
}

# Change these to your cruzid
$NAME = "kydiep"
$POD_NAME = "kydiep-carla"
$CACHE_NAME = "kydiep-carla-cache"

# Ensure an argument was provided
if ($args.Count -lt 1) {
    Write-Host "Usage: setup.ps1 {0=start/update, 1=stop, 2=delete including saved data}"
    exit 1
}

$mode = [int]$args[0]

if ($mode -eq 1) {
    Invoke-Kubectl delete deployment $NAME
    Invoke-Kubectl delete service $NAME
    Invoke-Kubectl delete ingress $NAME

} elseif ($mode -eq 0) {
    Invoke-Kubectl apply -f storage.yml
    Invoke-Kubectl apply -f cache.yml
    Invoke-Kubectl apply -f desktop.yml
    Invoke-Kubectl apply -f desktop-ingress.yml
    Invoke-Kubectl rollout status deployment/$NAME --timeout=300s
    Write-Host "Desktop ready: https://kydiep.nrp-nautilus.io/"

} elseif ($mode -eq 2) {
    Invoke-Kubectl delete deployment $NAME
    Invoke-Kubectl delete service $NAME
    Invoke-Kubectl delete ingress $NAME
    Invoke-Kubectl delete pvc --now $POD_NAME
    Invoke-Kubectl delete pvc --now $CACHE_NAME

} else {
    Write-Host "Usage: setup.ps1 {0=start/update, 1=stop, 2=delete including saved data}"
    exit 1
}
