FROM nvcr.io/nvidia/pytorch:22.03-py3
COPY requirments.txt /opt/app/requirments.txt
RUN pip install -r /opt/app/requirments.txt