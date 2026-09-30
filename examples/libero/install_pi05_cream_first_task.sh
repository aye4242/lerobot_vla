#!/usr/bin/env bash
set -euo pipefail

# Install the custom three-object task into the running Pi0.5 LIBERO container.
# The task must use this exact prompt because demonstrations are ordered
# cream cheese -> butter -> alphabet soup.
CONTAINER_NAME="${CONTAINER_NAME:-mujoco-huanghb-pi05}"
TASK_FILE="LIVING_ROOM_SCENE2_put_the_alphabet_soup_tomato_sauce_and_cream_cheese_box_in_the_basket.bddl"
TASK_PROMPT="put the cream cheese box, butter, and alphabet soup in the basket, one at a time"
PACKAGE_ROOT="/lerobot/.venv/lib/python3.12/site-packages/libero/libero"

docker cp "examples/libero/tasks/${TASK_FILE}" \
  "${CONTAINER_NAME}:${PACKAGE_ROOT}/bddl_files/libero_10/${TASK_FILE}"

docker exec -u 0 -i "${CONTAINER_NAME}" /lerobot/.venv/bin/python - "${PACKAGE_ROOT}/benchmark/__init__.py" "${TASK_PROMPT}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
prompt = sys.argv[2]
source = path.read_text()
new = f'language="{prompt}"'
old_prompts = (
    'language="put the alphabet soup, tomato sauce, and cream cheese box in the basket, one at a time"',
    'language="put the cream cheese box, alphabet soup, and tomato sauce in the basket, one at a time"',
)
for old in old_prompts:
    if old in source:
        source = source.replace(old, new, 1)
        break
else:
    if new not in source:
        raise RuntimeError(f"Could not find the Pi0.5 composition prompt override in {path}")
if new not in source:
    raise RuntimeError(f"Could not find the Pi0.5 composition prompt override in {path}")
path.write_text(source)
PY

docker exec -e LIBERO_CONFIG_PATH=/home/user_lerobot/.libero "${CONTAINER_NAME}" \
  /lerobot/.venv/bin/python -c "from libero.libero import benchmark; print(benchmark.get_benchmark_dict()['libero_10']().get_task(0).language)"
