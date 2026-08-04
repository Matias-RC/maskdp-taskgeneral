# Masked Desicion process Research
## File Descriptions

`conda_env.yml`: Conda YAML specifying Python & package dependencies. Used to create the "maskdp" conda environment.

```bash
conda env create -f conda_env.yml
conda activate maskdp
```

`datachecker.py`: File for checking the structure of a .npz data file in the dataset.

`dmc.py`: Wrapper for DeepMind Control

`agent/`: Contains agent definitions for MaskDP.

- `mdp`: Masked DP agent 

`custom_dmc_tasks/`: Contains custom DM Control tasks. 

## Design Choices

To resolve the scalability and readability challenges, this repository draws from the classic book *Design Patterns: Elements of Reusable Object-Oriented Software*.

Every architectural module contains its own `README.md`. These files map the local scripts to the classic design pattern catalog (Creational, Structural, Behavioral) or their modern ML variants.

**Example Structure:**
```
├── data/
    ├── __init__.py
    ├── README.md              # ← Explains data patterns used here
    ├── replay_buffer.py       # Behavioral: Sequence Rollout Buffer Iterator
    └── loaders.py             # Creational: Streaming Data Iterator
```