#!/bin/bash
# Patch cruz-control/carla-gym-env to work on the Nautilus desktop.
# Usage: bash carla_gym_fixes.sh ~/persistent/carla-gym-env
set -eu
cd "${1:-$HOME/persistent/carla-gym-env}"

# CARLA runs with -carla-rpc-port=4000. carla-v1 ignores params and hard-codes 2000.
sed -i 's/carla.Client("localhost", 2000)/carla.Client("localhost", 4000)/' gym_carla/envs/new_env.py
sed -i "s/'port': 2000/'port': 4000/" train.py run.py

# The desktop pod has one GPU.
sed -i 's/device="cuda:1"/device="cuda"/' train.py

# train.py trains on carla-v1 (3 actions); run.py must load the model into the same env.
sed -i "s/gym.make('carla-v0', params=params)/gym.make('carla-v1')/" run.py

# Log to a clearer folder name (the model is SAC, not DQN).
sed -i 's#tensorboard_log="./tensorboard_DQN/"#tensorboard_log="./tensorboard_SAC/"#' train.py run.py

# dash>=3 needs a newer typing_extensions than TensorFlow allows; open3d imports dash.
pip install -q "dash<3"

grep -n 'localhost' gym_carla/envs/new_env.py
grep -n "cuda\|carla-v\|tensorboard_log" train.py run.py
echo "carla-gym-env patched."

# A* route planner in reset() never terminates on some start/goal pairs: waypoint.next(2.0)
# gives a new waypoint id for every slightly different s, so nothing is ever "visited" and the
# open set grows until the pod is OOMKilled (32 GB). Add a visited set + expansion cap, and have
# reset() pick a new goal when the search gives up.
python - <<'PY'
p = "gym_carla/envs/new_env.py"
s = open(p).read()
if "max_expansions" not in s:
    reps = [
        ("    max_distance=5000,\n):",
         "    max_distance=5000,\n    max_expansions=20000,\n):"),
        ("    while not open_set.empty():\n        current_node = open_set.get()[2]\n",
         "    closed = set()\n"
         "    while not open_set.empty():\n        current_node = open_set.get()[2]\n"
         "        wp = current_node.waypoint\n"
         "        key = (wp.road_id, wp.section_id, wp.lane_id, round(wp.s))\n"
         "        if key in closed:\n            continue\n"
         "        closed.add(key)\n"
         "        if len(closed) > max_expansions:\n            break\n"),
        ("          self.route = a_star(self.world, start_waypoint, end_waypoint)\n",
         "          self.route = a_star(self.world, start_waypoint, end_waypoint)\n"
         "          if self.route is None:\n            continue\n"),
    ]
    for old, new in reps:
        assert s.count(old) == 1, "patch target not found: " + old[:50]
        s = s.replace(old, new)
    open(p, "w").write(s)
print("A* patch applied")
PY

# reset() spawned the ego car at start_waypoint.transform (road-surface height), which CARLA
# rejects as a collision, so the spawn loop never exits (and re-downloads the map each try,
# growing CARLA's memory). Spawn at the map spawn point instead (it sits just above the road).
sed -i 's/self.world.try_spawn_actor(self.ego_bp, start_waypoint.transform)/self.world.try_spawn_actor(self.ego_bp, point_a)/' gym_carla/envs/new_env.py
grep -n "try_spawn_actor(self.ego_bp" gym_carla/envs/new_env.py

# Memory: observations are 786,532 floats (4 x 256x256 RGB cameras), so MlpPolicy's default
# [256,256] net puts ~200M weights in each first layer; SAC's 5 networks + Adam + replay buffer
# need 20+ GB and the pod (32 GB, CARLA included) gets OOMKilled. Use a [64,64] net (~6 GB total).
sed -i 's/SAC("MlpPolicy", env, device="cpu", buffer_size=500, verbose=1,/SAC("MlpPolicy", env, device="cpu", buffer_size=500, policy_kwargs=dict(net_arch=[64, 64]), verbose=1,/' train.py
# run.py built a throwaway full-size SAC on the GPU (buffer 20000) before loading the saved one.
sed -i '/model = SAC("MlpPolicy", env, device="cuda", buffer_size=20000/d' run.py
sed -i 's/model = SAC.load("SAC_dist")/model = SAC.load("SAC_dist", device="cpu")/' run.py
grep -n "SAC(" train.py; grep -n "SAC" run.py
