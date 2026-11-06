# Settings for the scripts, overridable per machine in data/local.sh (not in the repo: it names the store's groups and
# local paths). Sourced from the repo root. data/local.sh can set:
#   VALL_E_STORE=<store root>              (default data/store)
#   GROUPS_=(<group> ...)                  the groups to train on (default: all of them)
#   VALL_E_TRAIN_PY=<python for training>  (default python)
#   VALL_E_EDIT_TEST=<edit / continue test script>
# a value already in the environment wins over data/local.sh's (e.g. VALL_E_TRAIN_PY=venv/bin/python for one run)
_env_py=$VALL_E_TRAIN_PY _env_store=$VALL_E_STORE
[ -f data/local.sh ] && . data/local.sh
VALL_E_TRAIN_PY=${_env_py:-$VALL_E_TRAIN_PY} VALL_E_STORE=${_env_store:-$VALL_E_STORE}
S=${VALL_E_STORE:-data/store}
GROUPS_ARGS=()
[ ${#GROUPS_[@]} -gt 0 ] && GROUPS_ARGS=(--groups "${GROUPS_[@]}")
