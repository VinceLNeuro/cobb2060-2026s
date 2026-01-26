#!/usr/bin/env bash

wd="/ihome/hpark/til177/GitHub/cobb2060-2026s/Data_cobb2060/proj1"

for url in \
    https://datahub.assets.cbioportal.org/brca_metabric.tar.gz \
    https://datahub.assets.cbioportal.org/brca_tcga.tar.gz \
    https://datahub.assets.cbioportal.org/coadread_tcga_pan_can_atlas_2018.tar.gz \
    https://datahub.assets.cbioportal.org/kirc_tcga.tar.gz \
    https://datahub.assets.cbioportal.org/lgg_tcga.tar.gz \
    https://datahub.assets.cbioportal.org/ucec_tcga_pan_can_atlas_2018.tar.gz \
    https://datahub.assets.cbioportal.org/hnsc_tcga.tar.gz \
    https://datahub.assets.cbioportal.org/luad_tcga.tar.gz \
    https://datahub.assets.cbioportal.org/thca_tcga.tar.gz \
    https://datahub.assets.cbioportal.org/lusc_tcga.tar.gz
do
    # wget -P ${wd} -c ${url}
    tar -xzvf ${wd}/$(basename ${url}) -C ${wd}
done

