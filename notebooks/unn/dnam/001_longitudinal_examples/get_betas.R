rm(list=ls())

###############################################
# Installing packages
###############################################
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("ChAMP")

library("ChAMP")

###############################################
# Setting variables
###############################################
path <- "E:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/raw/idat"
setwd(path)

arraytype <- 'EPIC'

###############################################
# Import and filtration
###############################################
myLoad <- champ.load(
  directory = path,
  arraytype = arraytype,
  method = "minfi",
  methValue = "B",
  autoimpute = TRUE,
  filterDetP = TRUE,
  ProbeCutoff = 0.1,
  SampleCutoff = 0.1,
  detPcut = 0.01,
  filterBeads = FALSE,
  beadCutoff = 0.05,
  filterNoCG = FALSE,
  filterSNPs = FALSE,
  filterMultiHit = FALSE,
  filterXY = FALSE,
  force = TRUE
)
pd <- as.data.frame(myLoad$pd)

###############################################
# Normalization
###############################################
betas_funnorm <- getBeta(preprocessFunnorm(myLoad$rgSet))
write.csv(betas_funnorm, file = "betas_funnorm.csv")
