FROM nvcr.io/nvidia/pytorch:21.06-py3
# FROM elyra/elyra:latest
#FROM nvidia/cuda:11.3-runtime-ubuntu18.04
# Set the locale
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y locales && \
    sed -i -e 's/# pt_PT ISO-8859-1/pt_PT ISO-8859-1/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales
ENV LANG pt_PT
ENV LANGUAGE pt_PT
ENV LC_ALL pt_PT

RUN apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.9 \
    python3.9-dev \
    python3.9-distutils \
    curl \
    gcc \
    make \
    vim \
    curl \ 
    git \
    sudo
RUN :\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :
RUN  update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN curl -sL https://deb.nodesource.com/setup_14.x \
    -o nodesource_setup.sh && bash ./nodesource_setup.sh && apt install nodejs -y

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
RUN pip3 install --upgrade pip \
    setuptools

# RUN pip3 install --upgrade torch==1.10.0+cu113 \
#    torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 torchtext==0.11.0 \
#    -f https://download.pytorch.org/whl/cu113/torch_stable.html
# RUN pip3 install torch-scatter torch-sparse \
#    torch-cluster torch-spline-conv torch-geometric \
#    -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN curl -sSL https://install.python-poetry.org | python3 - 
RUN echo "export PATH=$HOME/.local/bin:$PATH" >> $HOME/.bashrc
RUN pip3 install numpy==1.22.2 pandas==1.4.0 matplotlib==3.5.1 scikit-learn tqdm seaborn scipy \
    Pillow PyYAML jedi pytest pytest-black pytest-flake8  --no-cache-dir

RUN pip3 install ase==3.22.1 pymatgen==2022.2.7 nglview==3.0.3 networkx==2.6.3 joblib==1.1.0
RUN pip3 install --upgrade jupyterlab elyra[all]
#RUN pip3 install kubeflow-kale
RUN jupyter lab build --dev-build=False --minimize=False
RUN pip3 install --upgrade elyra-pipeline-editor-extension \
    elyra-code-snippet-extension elyra-code-viewer-extension \
    elyra-python-editor-extension
RUN jupyter labextension update --all
RUN pip3 install ipywidgets ipympl
RUN groupadd -g 1000 jovyan
RUN useradd -m -u 1000 -g 1000 -s /bin/bash jovyan
RUN usermod -aG sudo jovyan
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN echo 's3:x:1337:jovyan' >> /etc/group
USER jovyan
ENV NB_PREFIX /
CMD ["sh","-c", "jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 \
    --no-browser --allow-root --port=8888 --NotebookApp.token='' \
    --NotebookApp.password='' --NotebookApp.allow_origin='*' \
    --NotebookApp.base_url=${NB_PREFIX}"]
