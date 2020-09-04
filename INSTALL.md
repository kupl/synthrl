# Install SynthRL Locally
This file shows how to install SynthRL locally.

## Ubuntu
We have tested with following distributions.
* 18.04 LTS
* 20.04 LTS

### Requirements
First, install `python3` and `pip3`.
```bash
$ sudo apt update
$ sudo apt install -y python3 python3-pip
```
Install [PyTorch](https://pytorch.org)(>=1.4) that suits your system.
Other requirements can be installed with the following command.
```bash
$ sudo pip3 install -r requirement.txt
```

### Run
Download Synthrl and SynthRL benchmark.
```bash
$ cd /to/your/workspace
$ git clone https://github.com/kupl/synthrl.git
$ git clone https://github.com/kupl/synthrl-bench.git
```
The following command will execute short example.
```bash
$ python3 synthrl.py test \
            --setting ../synthrl-bench/bitvector/small.json \
            --synth SimpleSynthesizer --synth-func RandomFunction --synth-max-move 1000 \
            --veri SimpleVerifier --veri-func RandomFunction --testing RandomTesting --testing-args max_attempt=100 --veri-max-move 10000
```
To see more usage, type in the following command.
```bash
$ python3 synthrl.py -h
```
