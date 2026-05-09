#!/usr/bin/env bash

set -euo pipefail

SERVER="${1:-10.176.37.31}"
curl "http://${SERVER}:8080/list"
echo
