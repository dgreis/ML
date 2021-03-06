#Take base container
#FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-training:1.15.2-gpu-py36-cu100-ubuntu18.04
#
## Add custom stack of code
#RUN git clone https://github.com/dgreis/ML.git

FROM tensorflow/tensorflow:nightly-gpu

RUN apt-get update && apt-get install -y openssh-server

# Load an ssh server. Change root username and password. By default in debian, password login is prohibited,
# go into the file that controls this and make a change to allow password login
RUN mkdir /var/run/sshd
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN /etc/init.d/ssh restart

# Install git, so we can pull in some repos
RUN apt-get update && apt-get upgrade -y && apt-get install -y git
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y awscli

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# Install the requirements and the libraries we need (from a requirements.txt file)
RUN pip install 'numpy==1.18.2'
COPY requirements.txt /tmp/
RUN python3 -m pip install -r /tmp/requirements.txt

# These are local libraries, add them (assuming a setup.py)
#ADD your_libs_directory /your_libs_directory
#RUN python3 -m pip install /your_libs_directory
#RUN python3 your_libs_directory/setup.py install

# Adding git repos (optional - assuming a setup.py)
ENV AWS_CONFIG_FILE='~/.aws/config'

# Cleanup
RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]