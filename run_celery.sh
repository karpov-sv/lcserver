#!/bin/sh

watchmedo auto-restart -d lcserver -p '*.py' --ignore-patterns="*/.*" -- python3 -m celery -- -A lcserver worker --loglevel=info
