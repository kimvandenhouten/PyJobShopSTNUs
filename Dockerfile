FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y bash wget curl unzip build-essential \
                          python3 python3-pip openjdk-21-jdk \
    && rm -rf /var/lib/apt/lists/*

# 1) Install CPLEX silently
WORKDIR /opt/cplex_installer
COPY cplex_studio2212.linux_x86_64.bin installer.properties ./
RUN chmod +x cplex_studio2212.linux_x86_64.bin \
    && ./cplex_studio2212.linux_x86_64.bin -f installer.properties

# 2) Set CPLEX env vars
ENV CPLEX_STUDIO_DIR2212=/opt/ibm/ILOG/CPLEX_Studio2212
ENV PATH="${CPLEX_STUDIO_DIR2212}/cpoptimizer/bin/x86-64_linux:${PATH}"
ENV LD_LIBRARY_PATH="${CPLEX_STUDIO_DIR2212}/cpoptimizer/bin/x86-64_linux:${CPLEX_STUDIO_DIR2212}/cplex/lib"

# 3) Copy your code & install your Python packages
WORKDIR /workspace
COPY . /workspace
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
    && pip3 install -e .

# 4) Default to an interactive shell
CMD ["/bin/bash"]
