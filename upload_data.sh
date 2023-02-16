#!/bin/bash
THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
DATA="$THIS_SCRIPT_DIR/data"
RSYNC_PATH="$1"

rsync -chavzp --progress $DATA/*preprocessed.json "$RSYNC_PATH"