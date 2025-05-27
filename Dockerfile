# Use an official Ubuntu image
FROM --platform=linux/amd64 ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# 1) Install system dependencies (including python3-dev for building C extensions)
RUN apt-get update && apt-get install -y \
    bash \
    wget \
    curl \
    unzip \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    openjdk-21-jdk \
  && rm -rf /var/lib/apt/lists/*

# 2) Create installer working dir
WORKDIR /opt/cplex_installer

# 3) Copy your 22.1.1 binary and properties
COPY cplex_studio2211.linux_x86_64.bin installer.properties ./

# 4) Make installer executable and run silently
RUN chmod +x cplex_studio2211.linux_x86_64.bin && \
    ./cplex_studio2211.linux_x86_64.bin -f installer.properties -i silent

# 5) Install CPLEX Python bindings
RUN python3 -m pip install --upgrade pip && \
    python3 /opt/ibm/ILOG/CPLEX_Studio2211/python/setup.py install --user

# 6) Install DOcplex and scientific stack
RUN python3 -m pip install --user docplex numpy matplotlib pandas

# 7) Expose CP Optimizer on PATH
ENV PATH="/opt/ibm/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux:$PATH"

# 8) Convenience symlink
RUN ln -sf /opt/ibm/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux/cpoptimizer /usr/local/bin/cpoptimizer

# 9) Switch to your workspace
WORKDIR /workspace
ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
