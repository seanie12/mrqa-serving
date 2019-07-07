# File Name, settings etc.
DOCKER_NAME="adieujw/mrqa:1.0"
PORT_NUM="8888"

############################## RUN ###############################
### CodaLab arguments
CODALAB_ARGS="cl run"

# Name of bundle (can customize however you want)
CODALAB_ARGS="$CODALAB_ARGS --name run-predictions"
# Docker image (default: codalab/default-cpu)
CODALAB_ARGS="$CODALAB_ARGS --request-docker-image $DOCKER_NAME"
# Explicitly ask for a worker with at least one GPU
CODALAB_ARGS="$CODALAB_ARGS --request-gpus 1"
# Control the amount of RAM your run needs
CODALAB_ARGS="$CODALAB_ARGS --request-memory 11g"
# Kill job after this many days (default: 1 day)
CODALAB_ARGS="$CODALAB_ARGS --request-time 2d"

# Bundle dependencies
CODALAB_ARGS="$CODALAB_ARGS data_dir:mrqa-dev-data"
CODALAB_ARGS="$CODALAB_ARGS :predict_server.py"
CODALAB_ARGS="$CODALAB_ARGS :mrqa_official_eval.py"
CODALAB_ARGS="$CODALAB_ARGS :src"                                # Src folder (serve.py)
CODALAB_ARGS="$CODALAB_ARGS :config"                             # Config folder (parameters)

### Command to execute (these flags can be overridden) from the command-line
CMD="python3.6 src/serve.py $PORT_NUM & python3.6 predict_server.py <(cat data_dir/*.jsonl) predictions.json $PORT_NUM; for data_file in \`ls data_dir/*.jsonl\`; do base=\$(echo \${data_file} | cut -d \"/\" -f2); python3.6 mrqa_official_eval.py \${data_file} predictions.json > eval_\${base::-1}; done"

# Pass the command-line arguments through to override the above
# if [ -n "$1" ]; then
#   CMD="$CMD $@"
# fi

# Create the run on CodaLab!
FINAL_COMMAND="$CODALAB_ARGS \"$CMD\""
echo $FINAL_COMMAND
exec bash -c "$FINAL_COMMAND"