FROM ubuntu:22.04

# Define user and group information
ARG LDAP_USERNAME=tiowang
ARG LDAP_UID=268635
ARG LDAP_GROUPNAME=rcp-runai-ivrl
ARG LDAP_GID=30034

RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} && \
    useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

RUN apt update && apt install -y python3-pip

WORKDIR /home/${LDAP_USERNAME}
USER ${LDAP_USERNAME}

RUN pip install -r requirements.txt || true
RUN pip install matplotlib numpy scipy
