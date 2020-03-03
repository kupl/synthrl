# SynthRL [ˈsɪnθrɪl]
Program **Synth**esizer based on OGIS using learning-based agents trained with **R**einforcement **L**earning 

## Download
Clone this repository via SSH.
```
$ git clone git@github.com:kupl/SynthRL-dev.git
```
Or, via HTTPS.
```
$ git clone https://github.com/kupl/SynthRL-dev.git
```

## Run SynthRL using Docker
We provide a docker image to execute the program.

### Build
Move to project directory and build the image.
```
$ cd /path/to/SynthRL-dev
$ docker build -t synthrl .
```

### Run
Mount the volume that contains excution scripts and run the container.
```
$ docker run -it -v "$(pwd)/example:/example" synthrl python /example/list_language_test.py
```

## Run SynthRL Locally
If you prefer to install SynthRL to your own system, loot at the following instructions.

### Requirements
Currently, there are no required libraries and programs.

### Install SynthRL
Move to project directory and install using pip.
```
$ cd /path/to/SynthRL-dev
$ sudo pip install .
```

### Uninstall SyntrhRL
Uninstall using pip
```
$ sudo pip uninstall synthrl
```
