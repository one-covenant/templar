#!/usr/bin/env python3
"""Set 100% weights to burn UID every 360 blocks."""

import argparse
import time

import bittensor as bt

import tplr

BURN_UID = 1  # Same as validator.py
WEIGHT_INTERVAL = 360  # Blocks between weight sets


def get_config():
    parser = argparse.ArgumentParser(description="Burn-only validator script")
    parser.add_argument(
        "--netuid", type=int, default=268, help="Bittensor network UID."
    )
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    return bt.config(parser)


def main():
    config = get_config()

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)

    metagraph = subtensor.metagraph(config.netuid)
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    tplr.logger.info(f"Running burn on UID {my_uid}, netuid {config.netuid}")
    tplr.logger.info(
        f"Will set 100% weight to burn UID {BURN_UID} every {WEIGHT_INTERVAL} blocks"
    )

    last_set_block = 0
    log_interval = 10  # Log status every N iterations (~2 minutes)
    iteration = 0

    while True:
        try:
            current_block = subtensor.block
            blocks_since_last = current_block - last_set_block
            blocks_until_next = max(0, WEIGHT_INTERVAL - blocks_since_last)

            if blocks_since_last >= WEIGHT_INTERVAL:
                tplr.logger.info(
                    f"Block {current_block}: Setting weights to burn UID {BURN_UID}"
                )

                success, msg = subtensor.set_weights(
                    wallet=wallet,
                    netuid=config.netuid,
                    uids=[BURN_UID],
                    weights=[1.0],
                    wait_for_inclusion=True,
                )

                if success:
                    tplr.logger.info(
                        f"Weights set successfully at block {current_block}"
                    )
                    last_set_block = current_block
                else:
                    tplr.logger.warning(f"Failed to set weights: {msg}")
            else:
                # Periodic status log
                iteration += 1
                if iteration % log_interval == 0:
                    tplr.logger.info(
                        f"Block {current_block} | "
                        f"Blocks until next weight set: {blocks_until_next} | "
                        f"Last set: {last_set_block or 'never'}"
                    )

            time.sleep(12)

        except KeyboardInterrupt:
            tplr.logger.info("Shutting down...")
            break
        except Exception as e:
            tplr.logger.error(f"Error: {e}")
            time.sleep(12)


if __name__ == "__main__":
    main()
