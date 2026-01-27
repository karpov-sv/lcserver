#!/bin/sh

# Avoid crashing on newer MacOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

watchmedo auto-restart -d lcserver --recursive -p '**/*.py' -- python3 -m celery -- -A lcserver worker -P threads --loglevel=info
