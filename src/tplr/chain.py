# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# type: ignore

# Global imports
import asyncio
from collections import defaultdict
from types import SimpleNamespace

import bittensor as bt
import numpy as np
import torch
from bittensor import Wallet
from bittensor.core.chain_data import decode_account_id
from pydantic import ValidationError

# Local imports
from .logging import logger
from .schemas import Bucket


class ChainManager:
    """Base class for handling chain interactions."""

    def __init__(
        self,
        config,
        hparams=None,
        fetch_interval: int = 600,  # Fetch interval in seconds
        wallet: bt.wallet | None = None,
        bucket: Bucket | None = None,
    ):
        """
        Initialize chain commitment handler.

        Args:
            config: Bittensor config object.
            hparams: Hyperparameters namespace containing model configuration.
            fetch_interval (int): Interval in seconds between fetching commitments.
            wallet (bt.wallet, optional): Wallet to sign commitments.
            bucket (Bucket, optional): Bucket configuration to commit.
        """
        self.config = config
        self.netuid = self.config.netuid
        self.hparams = hparams or SimpleNamespace()

        # Bittensor objects
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.netuid)

        # Block and window tracking
        self.current_block = 0
        self.current_window = 0
        self.window_duration = self.hparams.blocks_per_window
        self.window_time = 0
        self.window_seeds = {}

        # Events
        self.block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()

        # Initialize bucket storage
        self.commitments = {}
        self.peers = []
        self.eval_peers = defaultdict(int)
        self.fetch_interval = fetch_interval
        self._fetch_task = None

        # Store wallet and bucket
        self.wallet = wallet
        self.bucket = bucket

    def start_commitment_fetcher(self):
        """Attach to the already-running event loop."""
        if self._fetch_task is None:
            self._fetch_task = asyncio.create_task(
                self._fetch_commitments_periodically()
            )

    async def _fetch_commitments_periodically(self):
        """Background task to periodically fetch commitments."""
        while True:
            try:
                # Create new subtensor instance for metagraph sync
                await asyncio.to_thread(
                    lambda: self.metagraph.sync(subtensor=self.subtensor)
                )

                # Create new subtensor instance for commitments
                commitments = await self.get_commitments()
                if commitments:
                    self.commitments = commitments
                    logger.debug(f"Updated commitments: {self.commitments}")
            except Exception as e:
                logger.error(f"Error fetching commitments: {e}")
                self.subtensor.substrate.close()
                self.subtensor.substrate.initialize()
            await asyncio.sleep(self.fetch_interval)

    def get_bucket(self, uid: int) -> Bucket | None:
        """Helper function to get the bucket for a given UID.

        Args:
            uid (int): The UID to retrieve the bucket for.

        Returns:
            Optional[Bucket]: The bucket corresponding to the UID, or None if not found.
        """
        return self.commitments.get(uid)

    def get_all_buckets(self) -> dict[int, Bucket | None]:
        """Helper function to get all buckets for all UIDs in the metagraph.

        Returns:
            Dict[int, Optional[Bucket]]: Mapping of UIDs to their bucket configurations
        """
        return {uid: self.get_bucket(uid) for uid in self.metagraph.uids}

    def commit(self, wallet: "bt.wallet", bucket: Bucket) -> None:
        """Commits bucket configuration to the chain.

        Args:
            wallet (bt.wallet): Wallet to sign the commitment
            bucket (Bucket): Bucket configuration to commit
        """
        concatenated = (
            bucket.account_id + bucket.access_key_id + bucket.secret_access_key
        )
        if self.netuid is not None:
            self.subtensor.commit(wallet, self.netuid, concatenated)
            logger.info(
                f"Committed bucket configuration to chain for hotkey {wallet.hotkey.ss58_address}"
            )

    def try_commit(self, wallet: Wallet, bucket: Bucket) -> None:
        """Attempts to verify existing commitment matches current bucket config and commits if not.

        Args:
            wallet (bt.wallet): Wallet to sign the commitment
            bucket (Bucket): Current bucket configuration to verify/commit
        """
        try:
            # Get existing commitment
            commitment = self.get_commitment(
                self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            )

            # Convert Bucket objects to concatenated strings for comparison
            commitment_str = (
                commitment.name
                + commitment.access_key_id
                + commitment.secret_access_key
            )
            bucket_details_from_env = (
                bucket.name + bucket.access_key_id + bucket.secret_access_key
            )

            logger.debug(
                "Comparing current commitment to bucket details from the environment:\n"
                f"Commitment: {commitment_str}\n"
                f"Current: {bucket_details_from_env}"
            )

            if bucket_details_from_env != commitment_str:
                if commitment_str == "":
                    log_msg_base = "Commitment is empty, likely because you're running your miner for the first time"
                else:
                    log_msg_base = "Bucket details have changed"
                logger.info(f"{log_msg_base}. Committing the new details.")
                self.commit(wallet, bucket)

        except Exception as e:
            logger.error(
                f"Error while verifying commitment: {str(e)}\n"
                "Committing the bucket details from the environment."
            )
            self.subtensor.substrate.close()
            self.subtensor.substrate.initialize()
            self.commit(wallet, bucket)

    def get_commitment(self, uid: int) -> Bucket:
        """Retrieves and parses committed bucket configuration data for a given
        UID.

        This method fetches commitment data for a specific UID from the
        subtensor network and decodes it into a structured format. The
        retrieved data is split into the following fields:
        - Account ID: A string of fixed length 32 characters.
        - Access key ID: A string of fixed length 32 characters.
        - Secret access key: A string of variable length (up to 64 characters).

        The parsed fields are then mapped to an instance of the `Bucket` class.
        When initializing the Bucket object, the account ID is also used as the
        bucket name.

        The retrieval process involves:
        - Fetching the commitment data for the specified UID using the
          configured `netuid` from the subtensor network.
        - Splitting the concatenated string into individual fields based on
          their expected lengths and order.
        - Mapping the parsed fields to a `Bucket` instance.

        **Note:** The order of fields (bucket name, account ID, access key ID,
        secret access key) in the concatenated string is critical for accurate
        parsing.

        Args:
            uid: The UID of the neuron whose commitment data is being
                retrieved.

        Returns:
            Bucket: An instance of the `Bucket` class containing the parsed
                bucket configuration details.

        Raises:
            ValueError: If the parsed data does not conform to the expected
                format for the `Bucket` class.
            Exception: If an error occurs while retrieving the commitment data
                from the subtensor network.
        """

        try:
            concatenated = self.subtensor.get_commitment(self.netuid, uid)
            logger.info(f"Commitment fetched: {concatenated}")
        except Exception as e:
            raise Exception(f"Couldn't get commitment from uid {uid} because {e}")
        if len(concatenated) != 128:
            raise ValueError(
                f"Commitment '{concatenated}' is of length {len(concatenated)} but should be of length 128."
            )

        try:
            return Bucket(
                name=concatenated[:32],
                account_id=concatenated[:32],
                access_key_id=concatenated[32:64],
                secret_access_key=concatenated[64:],
            )
        except ValidationError as e:
            raise ValueError(f"Invalid data in commitment: {e}")

    def decode_metadata(self, encoded_ss58: tuple, metadata: dict) -> tuple[str, str]:
        # Decode the key into an SS58 address.
        decoded_key = decode_account_id(encoded_ss58[0])
        # Get the commitment from the metadata.
        commitment = metadata["info"]["fields"][0][0]
        bytes_tuple = commitment[next(iter(commitment.keys()))][0]
        return decoded_key, bytes(bytes_tuple).decode()

    async def get_commitments(self, block: int | None = None) -> dict[int, Bucket]:
        """Retrieves all bucket commitments from the chain.

        Args:
            block (int, optional): Block number to query at

        Returns:
            Dict[int, Bucket]: Mapping of UIDs to their bucket configurations
        """
        try:
            substrate = self.subtensor.substrate
            # Query commitments via substrate.query_map
            query_result = substrate.query_map(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[self.netuid],
                block_hash=None if block is None else substrate.get_block_hash(block),
            )

            hotkey_to_uid = dict(zip(self.metagraph.hotkeys, self.metagraph.uids))
            commitments = {}

            for key, value in query_result:
                try:
                    decoded_ss58, commitment_str = self.decode_metadata(
                        key, value.value
                    )
                except Exception as e:
                    logger.error(f"Failed to decode metadata for key {key.value}: {e}")
                    continue

                if decoded_ss58 not in hotkey_to_uid:
                    continue

                uid = hotkey_to_uid[decoded_ss58]
                if len(commitment_str) != 128:
                    logger.error(
                        f"Invalid commitment length for UID {uid}: {len(commitment_str)}"
                    )
                    continue

                try:
                    bucket = Bucket(
                        name=commitment_str[:32],
                        account_id=commitment_str[:32],
                        access_key_id=commitment_str[32:64],
                        secret_access_key=commitment_str[64:],
                    )
                    commitments[uid] = bucket
                    logger.debug(f"Retrieved bucket commitment for UID {uid}")
                except Exception as e:
                    logger.error(f"Failed to build bucket for UID {uid}: {e}")
                    continue

            return commitments
        except Exception:
            self.subtensor.substrate.close()
            self.subtensor.substrate.initialize()
            return

    async def get_bucket_for_neuron(self, wallet: "bt.wallet") -> Bucket | None:
        """Get bucket configuration for a specific neuron's wallet

        Args:
            wallet (bt.wallet): The wallet to get bucket for

        Returns:
            Optional[Bucket]: The bucket assigned to this neuron, or None if not found
        """
        try:
            # Get UID by finding hotkey's index in metagraph
            uid = self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            return self.get_bucket(uid)
        except ValueError:
            logger.warning(
                f"Hotkey {wallet.hotkey.ss58_address} not found in metagraph"
            )
            return None

    async def fetch_commitments(self):
        """Fetches commitments and updates self.commitments."""
        commitments = await self.get_commitments()
        if commitments:
            self.commitments = commitments
            logger.debug(f"Fetched commitments: {self.commitments}")
        else:
            logger.warning("No commitments fetched.")

    def get_hotkey(self, uid: int) -> str | None:
        """Returns the hotkey for a given UID."""
        # Handle different data types for uids
        if isinstance(self.metagraph.uids, (np.ndarray, torch.Tensor)):
            uids_list = self.metagraph.uids.tolist()
        else:
            uids_list = self.metagraph.uids

        # Handle different data types for hotkeys
        if isinstance(self.metagraph.hotkeys, (np.ndarray, torch.Tensor)):
            hotkeys_list = self.metagraph.hotkeys.tolist()
        else:
            hotkeys_list = self.metagraph.hotkeys

        if uid in uids_list:
            index = uids_list.index(uid)
            return hotkeys_list[index]
        else:
            return None

    def update_peers_with_buckets(self):
        """Updates peers for gradient gathering, evaluation peers, and tracks inactive peers."""
        # Create mappings
        uid_to_stake = dict(
            zip(self.metagraph.uids.tolist(), self.metagraph.S.tolist())
        )

        # Get currently active peers
        active_peers = set(int(uid) for uid in self.active_peers)

        # Track inactive peers (previously active peers that are no longer active)
        previously_active = set(
            self.eval_peers.keys()
        )  # since self.eval_peers is now a dict
        newly_inactive = previously_active - active_peers
        self.inactive_peers = newly_inactive

        logger.debug(f"Active peers: {active_peers}")
        logger.info(f"Newly inactive peers: {newly_inactive}")
        logger.debug(f"Stakes: {uid_to_stake}")

        if not active_peers:
            logger.warning("No active peers found. Skipping update.")
            return

        # ---------------------------------------------------------------
        # Convert self.eval_peers into a dict while retaining old counts
        # for peers still active with stake <= 1000.
        # ---------------------------------------------------------------
        self.eval_peers = {
            int(uid): self.eval_peers.get(int(uid), 1)
            for uid in active_peers
            if uid in uid_to_stake and uid_to_stake[uid] <= 20000
        }

        logger.debug(f"Filtered eval peers: {list(self.eval_peers.keys())}")

        logger.info(f"Total evaluation peers: {len(self.eval_peers)}")
        logger.info(f"Total inactive peers: {len(self.inactive_peers)}")
