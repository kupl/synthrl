# SynthRL [ˈsɪnθrɪl]
Program **Synth**esizer based on OGIS using learning-based agents trained with **R**einforcement **L**earning 

## Docker
We provide a docker image to execute SynthRL.

### Build
Download the project and build the image.
```bash
$ git clone https://github.com/kupl/synthrl.git
$ cd synthrl
$ docker build . -t synthrl
```

### Run
The following command will execute short example.
```bash
$ docker run -t --rm synthrl python synthrl.py test \ 
          --setting bench/bitvector/small.json \
          --synth SimpleSynthesizer --synth-func RandomFunction --synth-max-move 1000 \ 
          --veri SimpleVerifier --veri-func RandomFunction --testing RandomTesting --testing-args max_attempt=100 --veri-max-move 10000
```
To see more usage, type in the following command.
```bash
$ docker run -t --rm synthrl python synthrl.py -h
``` 

## Install SynthRL Locally
Please see [INSTALL.md](INSTALL.md).
