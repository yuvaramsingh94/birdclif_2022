FROM nvcr.io/nvidia/tensorflow:21.10-tf2-py3
COPY requirments.txt /opt/app/requirments.txt
RUN pip install -r /opt/app/requirments.txt
RUN apt update
RUN apt-get install -y libsndfile1 