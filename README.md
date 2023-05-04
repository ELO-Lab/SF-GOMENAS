# Accelerating Gene-pool Optimal Mixing Evolutionary Algorithm for Neural Architecture Search with Synaptic Flow
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)

Khoa Huu Tran, Luc Truong, An Vo, and Ngoc Hoang Luong

## Setup
- Clone this repository
- Install packages
```
$ pip install -r requirements.txt
```
- Download [NATS-Bench](https://drive.google.com/drive/folders/17S2Xg_rVkUul4KuJdq0WaWoUuDbo8ZKB), put it in the `benchmark` folder and follow instructions [here](https://github.com/D-X-Y/NATS-Bench)
## Usage
To run the code, use the command below with the required arguments

```shell
python search.py --method <method_name> --dataset <dataset_name> --pop_size <population_size> --n_gens <number_of_generations> --n_runs <number_of_runs> 
```

Refer to `search.py` for more details.
Example commands:
```shell
# SF-GOMENAS
python search.py --method SF-GOMENAS --dataset cifar10 --pop_size 20 --n_gens 50 --n_runs 30
```


## Acknowledgement
Our source code is inspired by:

- [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://github.com/D-X-Y/NATS-Bench)
- [Zero-Cost Proxies for Lightweight NAS](https://github.com/SamsungLabs/zero-cost-nas)
- [More Concise and Robust Linkage Learning by Filtering and Combining Linkage Hierarchies](https://homepages.cwi.nl/~bosman/source_code.php)
- [Training-Free Multi-Objective and Many-Objective Evolutionary Neural Architecture Search with Synaptic Flow](https://github.com/ELO-Lab/TF-MaOENAS)
