// ecosystem.config.js
require('dotenv').config({ path: '.env' });

const { execSync } = require('child_process');
const RANDOM_SUFFIX = execSync(
  "cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1"
)
  .toString()
  .trim();

const PROJECT_NAME = `test_${RANDOM_SUFFIX}`;

module.exports = {
  apps: [
    /*───────────────────────── Miner ─────────────────────────*/
    {
      name            : "TM1",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
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
        CUDA_VISIBLE_DEVICES: "1,2"
      }
    },
    {
      name            : "TM2",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
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
        CUDA_VISIBLE_DEVICES: "3,4"
      }
    },

    /*──────────────────────── Validator ──────────────────────*/
    {
      name            : "TV1",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "1",
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
        CUDA_VISIBLE_DEVICES: "0"
      }
    }
  ]
};
