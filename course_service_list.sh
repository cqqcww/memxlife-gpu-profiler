#!/usr/bin/env bash

set -euo pipefail

SERVER="${1:-10.176.37.31}"
COURSE_CURL_CONNECT_TIMEOUT="${COURSE_CURL_CONNECT_TIMEOUT:-8}"
COURSE_CURL_MAX_TIME="${COURSE_CURL_MAX_TIME:-30}"
curl --connect-timeout "${COURSE_CURL_CONNECT_TIMEOUT}" --max-time "${COURSE_CURL_MAX_TIME}" -sS "http://${SERVER}:8080/list"
echo
