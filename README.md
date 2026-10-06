# ChronoChart

This is the code for the paper ChronoChart: Setup-Agnostic Channel Charting for
Label-Free Indoor Localization.
It is a setup-agnostic channel charting approach that only needs timestamp.


## Requirements

- Python 3.12
- CUDA-capable environment for training and prediction
- Dataset-specific files described below

Poetry manages the package metadata and locked core dependencies:

```bash
poetry install
```

The repository also maintains `requirements.txt`, which includes supplemental
runtime libraries used by Dichasus preprocessing
. At present, `requirements.txt` and the Poetry dependency table are
not identical. For a full training environment, create an isolated environment
and install the supplemental requirements as needed:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Filesystem configuration

All data and outputs are resolved through `localization/config/paths.py`.
Set `ROOT_DIR` before running a command if the default is not appropriate:

```bash
export ROOT_DIR=/path/to/localization-runtime
```

If unset, the root defaults to `/var/tmp/localization`. Expected runtime paths
include:

```text
$ROOT_DIR/data/
$ROOT_DIR/data/pt/
$ROOT_DIR/lightning_logs/
$ROOT_DIR/reports/
```

## Paper reproduction
To quickly run the experiments from the paper, run the noteboooks
`examples/train_5g.ipynb` for Fraunhofer 5G and `examples/train_dichasus.ipynb` for Dichasus.
Passive UWB will be added once the datset is public.

Due to problems with the original seed, this will not produce the exact values reported in the paper, but pretty close ones.

## Citation

If you use our work, please consider citing
```
@inproceedings{foroushani2026chrono,
    title={ChronoChart: Setup-Agnostic Channel Charting for Label-Free Indoor Localization},
    author={Sharghi Foroushani, Majid and Pirkl, Jonas and Ott, Jonathan and Stahlke, Maximilian and Yammine, George and Feigl, Tobias and Mutschler, Christopher},
    booktitle={2026 16th International Conference on Indoor Positioning and Indoor Navigation (IPIN)},
    pages={1--6},
    year={2026},
    organization={IEEE}
}
```

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
