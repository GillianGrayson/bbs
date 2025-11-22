rm(list=ls())


if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")

BiocManager::install("ChAMP")
BiocManager::install("ChAMPdata")
BiocManager::install("IlluminaHumanMethylation450kanno.ilmn12.hg19")
BiocManager::install("IlluminaHumanMethylationEPICanno.ilm10b4.hg19")
BiocManager::install("IlluminaHumanMethylationEPICv2anno.20a1.hg38")
BiocManager::install("org.Hs.eg.db")
BiocManager::install("IlluminaHumanMethylationEPICmanifest")
BiocManager::install("IlluminaHumanMethylation450kmanifest")
BiocManager::install("IlluminaHumanMethylationEPICv2manifest")
BiocManager::install("DMRcate")
BiocManager::install("geneLenDataBase")
BiocManager::install("GO.db")
install.packages("devtools")
devtools::install_version("kpmt",version="0.1.0")

library(ChAMP)
