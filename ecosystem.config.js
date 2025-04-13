require('dotenv').config({ path: '.env' });
const RANDOM_SUFFIX = require('child_process').execSync("cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1").toString().trim();
const PROJECT_NAME = `test_${RANDOM_SUFFIX}`;

module.exports = {
    apps: [
        {
            name: "TM1",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name templar_test --wallet.hotkey M1 --device cuda:1 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TM2",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name templar_test --wallet.hotkey M2 --device cuda:2 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TM3",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name templar_test --wallet.hotkey M3 --device cuda:3 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TM4",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name templar_test --wallet.hotkey M4 --device cuda:4 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TM5",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name templar_test --wallet.hotkey M5 --device cuda:5 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TM6",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name templar_test --wallet.hotkey M6 --device cuda:6 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TV1",
            script: "neurons/validator.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name templar_test --wallet.hotkey V1 --device cuda:0 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "Aggregator",
            script: "neurons/aggregator.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: "--netuid 3 --device cuda:7 --project templar"
        }
    ]
}
