# Use an official Ubuntu image
FROM ubuntu:22.04

# Install required dependencies
RUN apt-get update && apt-get install -y \
    bash \
    wget \
    curl \
    unzip \
    build-essential \
    python3 \
    python3-pip \
    openjdk-21-jdk \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive

# Create working directory
WORKDIR /opt/cplex_installer

# Copy your CPLEX installer + silent install config
COPY cplex_studio2212.linux_x86_64.bin .
COPY installer.properties .

# Make it executable and install silently
RUN chmod +x cplex_studio2212.linux_x86_64.bin && \
    ./cplex_studio2212.linux_x86_64.bin -f installer.properties

# Set PATH so Python/CP scripts can find cpoptimizer
ENV PATH="/opt/ibm/ILOG/CPLEX_Studio2212/cpoptimizer/bin/x86-64_linux:$PATH"

# Optional: install scientific Python tools
RUN pip3 install numpy matplotlib pandas

# Optional: add symlink for legacy scripts
RUN ln -s /opt/ibm/ILOG/CPLEX_Studio2212/cpoptimizer/bin/x86-64_linux/cpoptimizer /usr/local/bin/cpoptimizer

WORKDIR /workspace

CMD ["/bin/bash"]
