// ecosystem.config.js
require('dotenv').config({ path: '.env' });

const { execSync } = require('child_process');
const RANDOM_SUFFIX = execSync(
  "cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1"
)
  .toString()
  .trim();

const PROJECT_NAME = `2.1.9+test.tp.2b`;

module.exports = {
  apps: [
    /*───────────────────────── Miner ─────────────────────────*/
    {
      name            : "TM1",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : ".venv/bin/torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "2",
        "neurons/miner.py",
        "--wallet.name", "templar_test",
        "--wallet.hotkey", "M1",
        "--device", "cuda",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        DP_SHARD: "1",
        TP_DEGREE: "2",
        CUDA_VISIBLE_DEVICES: "0,1",
        TMPDIR: "/ephemeral/tmp",
        TEMP: "/ephemeral/tmp",
        TMP: "/ephemeral/tmp"
      }
    },
    {
      name            : "TM2",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : ".venv/bin/torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "2",
        "neurons/miner.py",
        "--wallet.name", "templar_test",
        "--wallet.hotkey", "M2",
        "--device", "cuda",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        DP_SHARD: "1",
        TP_DEGREE: "2",
        CUDA_VISIBLE_DEVICES: "2,3",
        TMPDIR: "/ephemeral/tmp",
        TEMP: "/ephemeral/tmp",
        TMP: "/ephemeral/tmp"
      }
    },
    {
      name            : "TM3",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : ".venv/bin/torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "2",
        "neurons/miner.py",
        "--wallet.name", "templar_test",
        "--wallet.hotkey", "M3",
        "--device", "cuda",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        DP_SHARD: "1",
        TP_DEGREE: "2",
        CUDA_VISIBLE_DEVICES: "4,5",
        TMPDIR: "/ephemeral/tmp",
        TEMP: "/ephemeral/tmp",
        TMP: "/ephemeral/tmp"
      }
    },

    /*──────────────────────── Validator ──────────────────────*/
    {
      name            : "TV1",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : ".venv/bin/torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "2",
        "neurons/validator.py",
        "--wallet.name", "templar_test",
        "--wallet.hotkey", "V1",
        "--device", "cuda",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        DP_SHARD: "1",
        TP_DEGREE: "2",
        CUDA_VISIBLE_DEVICES: "6,7",
        MAX_CATCHUP_WINDOWS: "5",
        TMPDIR: "/ephemeral/tmp",
        TEMP: "/ephemeral/tmp",
        TMP: "/ephemeral/tmp"
      }
    }
  ]
};