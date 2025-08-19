rm(list=ls())


if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("DMRcate")
BiocManager::install("methylumi")
BiocManager::install("ChAMP")
BiocManager::install("minfi")
BiocManager::install("minfiData")
BiocManager::install("wateRmelon")
BiocManager::install("shinyMethyl")
BiocManager::install("FlowSorted.Blood.EPIC")
BiocManager::install("FlowSorted.Blood.450k")
BiocManager::install("FlowSorted.DLPFC.450k")

library(ChAMP)
library(ChAMPdata)
library("xlsx")
library(openxlsx)
library(minfi)
library(minfiData)
library(shinyMethyl)
library(wateRmelon)
library(FlowSorted.Blood.EPIC)
library(FlowSorted.Blood.450k)
library(FlowSorted.DLPFC.450k)
library(minfiData)
library(sva)

path <- "E:/YandexDisk/DNAm draft/Lesnoy_CVD/GSE220622/raw"
setwd(path)

arraytype <- "EPIC"
detPcut <- 0.01

###############################################
# Import and filtration
###############################################
myLoad <- champ.load(
  directory = path,
  arraytype = arraytype, # Choose microarray type is "450K" or "EPIC".(default = "450K")
  method = "minfi", # Method to load data, "ChAMP" method is newly provided by ChAMP group, while "minfi" is old minfi way.(default = "ChAMP")
  methValue = "B", # Indicates whether you prefer m-values M or beta-values B. (default = "B")
  autoimpute = TRUE, # If after filtering (or not do filtering) there are NA values in it, should impute.knn(k=3) should be done for the rest NA?
  filterDetP = TRUE, # If filter = TRUE, then probes above the detPcut will be filtered out.(default = TRUE)
  ProbeCutoff = 0.1, # The NA ratio threshhold for probes. Probes with above proportion of NA will be removed.
  SampleCutoff = 0.1, # The failed p value (or NA) threshhold for samples. Samples with above proportion of failed p value (NA) will be removed.
  detPcut = detPcut, # The detection p-value threshold. Probes about this cutoff will be filtered out. (default = 0.01)
  filterBeads = FALSE, # If filterBeads=TRUE, probes with a beadcount less than 3 will be removed depending on the beadCutoff value.(default = TRUE)
  beadCutoff = 0.05, # The beadCutoff represents the fraction of samples that must have a beadcount less than 3 before the probe is removed.(default = 0.05)
  filterNoCG = TRUE, # If filterNoCG=TRUE, non-cg probes are removed.(default = TRUE)
  filterSNPs = TRUE, # If filterSNPs=TRUE, probes in which the probed CpG falls near a SNP as defined in Nordlund et al are removed.(default = TRUE)
  filterMultiHit = TRUE, # If filterMultiHit=TRUE, probes in which the probe aligns to multiple locations with bwa as defined in Nordlund et al are removed.(default = TRUE)
  filterXY = FALSE, # If filterXY=TRUE, probes from X and Y chromosomes are removed.(default = TRUE)
  force = TRUE
)
pd <- as.data.frame(myLoad$pd)

###############################################
# CpGs selection
###############################################
cpgs_fltd <- rownames(myLoad$beta)
write.csv(cpgs_fltd, file = "cpgs_fltd.csv", row.names=FALSE, col.names=FALSE)