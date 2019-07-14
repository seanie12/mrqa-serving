### CodaLab arguments
CODALAB_ARGS="cl run"

# Bundle dependencies
CODALAB_ARGS="$CODALAB_ARGS data_dir:mrqa-dev-data"
CODALAB_ARGS="$CODALAB_ARGS :run-predictions"

CMD="for data_file in \`ls data_dir/*.jsonl\`; do base=\$(echo \${data_file} | cut -d \"/\" -f2); python3.6 mrqa_official_eval.py \${data_file} run-predictions/predictions.json > eval_\${base::-1}; done"

# Create the run on CodaLab!
FINAL_COMMAND="$CODALAB_ARGS \"$CMD\""
echo $FINAL_COMMAND
exec bash -c "$FINAL_COMMAND"