#!/usr/bin/env python3
"""
Main entry point for running Yuanyou2 robot with OpenPI policy.

Usage:
    # Start policy server first:
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_yuanyou2_lora_finetune \
        --policy.dir=checkpoints/pi05_yuanyou2_lora_finetune/my_experiment/20000

    # Then run this script on the robot side:
    python examples/yuanyou2/main.py --remote-host 127.0.0.1
"""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.yuanyou2 import yuanyou2_environment as _env


@dataclasses.dataclass
class Args:
    """Command-line arguments for Yuanyou2 deployment."""

    remote_host: str = "127.0.0.1"
    """IP address of the policy server."""

    remote_port: int = 8000
    """Port of the policy server."""

    control_frequency: float = 10.0
    """Control loop frequency in Hz."""

    action_horizon: int = 30
    """Number of actions in each chunk returned by policy."""

    open_loop_horizon: int = 10
    """Number of actions to execute before querying policy again."""

    prompt: str = "pick a cube and place it on another cube"
    """Language instruction for the robot."""

    num_episodes: int = 100
    """Number of episodes to run."""

    max_episode_steps: int = 250
    """Maximum steps per episode."""


def main(args: Args) -> None:
    logging.info("=" * 60)
    logging.info("Yuanyou2 OpenPI Deployment")
    logging.info("=" * 60)
    logging.info(f"Policy server: ws://{args.remote_host}:{args.remote_port}")
    logging.info(f"Control frequency: {args.control_frequency} Hz")
    logging.info(f"Action horizon: {args.action_horizon} steps")
    logging.info(f"Open-loop horizon: {args.open_loop_horizon} steps")
    logging.info(f"Prompt: '{args.prompt}'")
    logging.info("=" * 60)

    if args.open_loop_horizon > args.action_horizon:
        logging.warning(
            f"open_loop_horizon ({args.open_loop_horizon}) > action_horizon ({args.action_horizon}). "
            "The policy may be queried before the previous chunk is exhausted."
        )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
    )

    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Connected to policy server. Metadata: {metadata}")

    environment = _env.Yuanyou2Environment(
        prompt=args.prompt,
    )

    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.open_loop_horizon,
        )
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],
        max_hz=args.control_frequency,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    logging.info("Starting Yuanyou2 robot control loop...")
    logging.info("Press Ctrl+C to stop.")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("Stopping robot. Ctrl+C pressed.")
    finally:
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    args: Args = tyro.cli(Args)
    main(args)