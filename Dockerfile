FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ADD ./requirements.txt /workspace/requirements.txt
WORKDIR /workspace
RUN pip install -r requirements.txt