import os
import time
from dataclasses import dataclass
import importlib
import sys

import numpy as np
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
import art
import emoji

import jax
from jax import numpy as jnp
import functools

from brax.io import html
import brax.envs as brax_envs
from brax.io import model as brax_model

import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig
from sac_mppi.brax_rl.brax_env import UnitreeGo2EnvRL, UnitreeH1EnvRL, UnitreeGo2DialEnvRL
from brax.training.agents.sac import networks as sac_networks

plt.style.use("science")

# Tell XLA to use Triton GEMM
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

def main():
    art.tprint("LeCAR @ CMU\nSAC-DIALMPC", font="big", chr_ignore=True)

    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    config_or_example.add_argument("--list-examples", action="store_true")
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    args = parser.parse_args()

    if args.list_examples:
        print("Examples:")
        for example in examples:
            print(f"  {example}")
        return

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    rng = jax.random.PRNGKey(seed=dial_config.seed)

    # Load SAC policy and value parameters
    normalizer_params_policy, policy_params = brax_model.load_params(config_dict['policy_path'])
    normalizer_params_value, q_params = brax_model.load_params(config_dict['value_path'])
    normalizer_params = normalizer_params_policy

    print(emoji.emojize(":rocket:") + " Creating environment")
    env = UnitreeGo2EnvRL(
        deploy=True,
        action_scale=1.0,
        dial_action_space=True,
        get_value=False
    )
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)

    obs_size = env.observation_size
    action_size = env.action_size

    def normalize_fn(obs, normalizer_params):
        return obs  # If needed, apply normalization as done in training

    sac_network = sac_networks.make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        hidden_layer_sizes=(512, 256,128),
    )

    # make_inference_fn returns a function that first takes (normalizer_params, policy_params)
    # and returns a policy function that takes (obs, rng).
    policy_factory = sac_networks.make_inference_fn(sac_network)
    policy_inference = policy_factory((normalizer_params, policy_params))

    rng, rng_policy = jax.random.split(rng)
    rng, rng_reset = jax.random.split(rng)
    state = reset_env(rng_reset)

    Nstep = dial_config.n_steps
    rews = []
    rollout = []
    observations = []
    actions = []

    print("SAC Inference Rollout:")
    with tqdm(range(Nstep), desc="SAC Inference Rollout") as pbar:
        for t in pbar:
            obs = state.obs
            rng_policy, rng_step = jax.random.split(rng_policy)
            # Now we call policy_inference(obs, rng_step)
            action = policy_inference(obs, rng_step)
            action = action[0]
            state = step_env(state, action)

            rollout.append(state.pipeline_state)
            rews.append(state.reward)
            observations.append(state.obs)
            actions.append(action)
            pbar.set_postfix({"rew": f"{state.reward:.2e}"})

    mean_rew = jnp.array(rews).mean()
    print(f"Mean reward = {mean_rew:.2e}")

    # Create output directory if not exists
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    observations = jnp.array(observations)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_observations"), observations)

    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
    )

    with open(
        os.path.join(dial_config.output_dir, f"{timestamp}_brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    data = []
    for i, pipeline_state in enumerate(rollout):
        data.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    pipeline_state.qpos,
                    pipeline_state.qvel,
                    pipeline_state.ctrl,
                ]
            )
        )
    data = jnp.array(data)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), data)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main()
