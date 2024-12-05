FROM python:3.13

ARG pip_installer="https://bootstrap.pypa.io/get-pip.py"
ARG awscli_version="1.16.236"

# install command.
RUN apt-get update && apt-get install -y less vim jq unzip sshpass

# create local bin directory
RUN mkdir -p /root/.local/bin
ENV PATH $PATH:/root/.local/bin

# install aws-cli
#RUN pip install --user --upgrade awscli==${awscli_version}

# install aws-cli ver2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

# install sam
RUN pip install --user --upgrade aws-sam-cli

# install eksctl
RUN curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
RUN mv /tmp/eksctl /usr/local/bin
RUN eksctl version

# install kubectl 1.17.9
RUN curl -o kubectl https://amazon-eks.s3.us-west-2.amazonaws.com/1.17.9/2020-08-04/bin/linux/amd64/kubectl
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin
RUN mkdir -p $HOME/bin && cp /usr/local/bin/kubectl $HOME/bin/kubectl && export PATH=$PATH:$HOME/bin
RUN echo 'export PATH=$PATH:$HOME/bin' >> ~/.bash_profile

RUN aws --version
RUN kubectl version --short --client

RUN echo "alias ll='ls -al'" >> /root/.bashrc

RUN pip install ansible

RUN ansible-galaxy collection install community.kubernetes

# user add
RUN groupadd -g 1000 user
RUN useradd -m -s /bin/bash -u 1000 -g 1000 yutarooo216
USER yutarooo216
WORKDIR /home/yutarooo216
