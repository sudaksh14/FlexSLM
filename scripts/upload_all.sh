#!/usr/bin/bash

PY_FILES=$(find . -name "*.py")
SH_FILES=$(find . -name "*.sh")

tar -czvf all.tar.gz $PY_FILES $SH_FILES
scp all.tar.gz $SNELLIUSREMOTE:~
rm all.tar.gz
ssh $SNELLIUSREMOTE "tar -xzvf all.tar.gz; rm all.tar.gz"
