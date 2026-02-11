#!/usr/bin/env bash

DEST="/ihome/hpark/til177/GitHub/cobb2060-2026s/Data_cobb2060/proj3"
mkdir -p ${DEST}
URL="https://bits.csb.pitt.edu/mscbio2066/assign3/data/"

# -np: no parent
# -nH: no hostname in directories (bits.csb.pitt.edu), -cut-dirs=3: cut first 3 dirs (/cobb2060/assign2/data) -> chr10
wget -r  -np  -nH --cut-dirs=3  -R "index.html*"  -P "${DEST}"  "${URL}"