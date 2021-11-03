FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install vim git -y

RUN /opt/conda/bin/conda install jupyter scipy matplotlib tqdm scikit-learn pandas -y \
  && /opt/conda/bin/pip install pyDOE \
  && /opt/conda/bin/pip install sobol_seq \
  && /opt/conda/bin/pip install torchdiffeq \
  && /opt/conda/bin/pip install fire \
  && /opt/conda/bin/pip install torchnet \
  && /opt/conda/bin/pip install git+https://github.com/AdamCobb/hamiltorch
