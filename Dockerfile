FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

# Working directory.
WORKDIR /src

# Library.
COPY ./requirement.txt .
RUN pip install -r requirement.txt

# Copy src.
COPY . .
RUN sed -i '1s/^/#!\/opt\/conda\/bin\/python\n/' synthrl.py
RUN chmod +x synthrl.py
RUN ln -s /src/synthrl.py /usr/bin/synthrl

# Environment setting.
COPY ./docker/bash.bashrc /etc/bash.bashrc

# Entry point
WORKDIR /workspace
# RUN git clone https://github.com/kupl/synthrl-bench.git bench
CMD ["synthrl", "-h"]
