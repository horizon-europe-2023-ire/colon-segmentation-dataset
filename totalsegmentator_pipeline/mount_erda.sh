#!/bin/bash
key=/home/smp884/.ssh/id_rsa      # path to your private key for ERDA
user=smp884@alumni.ku.dk          #your ERDA username
erdadir=/IRE-DATA                 # ERDA directory you want to mount
mnt=/home/smp884/IRE/data         # mount point on the cluster

if [ -f "$key" ]; then
    mkdir -p ${mnt}
    sshfs ${user}@io.erda.dk:${erdadir} ${mnt} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${key}
else
    echo "SSH key not found: ${key}"
fi
