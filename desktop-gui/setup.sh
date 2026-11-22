#!/bin/sh
set -eu
cd "$(dirname "$0")"

NAME=kydiep # Change this to your cruzid
POD_NAME=kydiep-carla # Change this to your cruzid
CACHE_NAME=kydiep-carla-cache # Change this to your cruzid

if [ "${1:-}" = 1 ]; then
    kubectl delete deployment $NAME
    kubectl delete service $NAME
    kubectl delete ingress $NAME
elif [ "${1:-}" = 0 ]; then
    kubectl apply -f storage.yml
    kubectl apply -f cache.yml
    kubectl apply -f desktop.yml
    kubectl apply -f desktop-ingress.yml
    kubectl rollout status deployment/$NAME --timeout=300s
    echo "Desktop ready: https://kydiep.nrp-nautilus.io/"
elif [ "${1:-}" = 2 ]; then
    kubectl delete deployment $NAME
    kubectl delete service $NAME
    kubectl delete ingress $NAME
    kubectl delete pvc --now $POD_NAME
    kubectl delete pvc --now $CACHE_NAME
else
    echo "Usage: $0 {0=start/update, 1=stop, 2=delete including saved data}"
    exit 1
fi
