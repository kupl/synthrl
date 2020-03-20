FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

COPY . /src/synthrl
RUN cd /src/synthrl; pip install .

CMD bash
