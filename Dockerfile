FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

# Working directory.
WORKDIR /src

# Library.
COPY ./requirement.txt .
RUN pip install -r requirement.txt

# Copy src.
COPY . .

# Environment setting.
COPY ./docker/bash.bashrc /etc/bash.bashrc

# Entry point
WORKDIR /workspace
RUN git clone https://github.com/kupl/synthrl-bench.git bench
CMD ['python', 'synthrl.py', '-h']
