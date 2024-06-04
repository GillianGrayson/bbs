rm(list=ls())

###############################################
# Installing packages
###############################################
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("ChAMP")
BiocManager::install("methylGSA")
BiocManager::install("preprocessCore")
install.packages("devtools")
install.packages("splitstackshape")
devtools::install_github("danbelsky/DunedinPACE")
devtools::install_github("https://github.com/regRCPqn/regRCPqn")
library("ChAMP")
library("preprocessCore")
library("DunedinPACE")
library("regRCPqn")
library(readxl)
library(splitstackshape)

###############################################
# Setting variables
###############################################
arraytype <- '450K'
dataset <- 'GSE142512'

###############################################
# Setting path
###############################################
path_data <- "D:/YandexDisk/Work/pydnameth/datasets/GPL13534/GSE142512/raw/idat"
path_pc_clocks <- "D:/YandexDisk/Work/pydnameth/datasets/lists/cpgs/PC_clocks/"
path_horvath <- "D:/YandexDisk/Work/pydnameth/draft/10_MetaEPIClock/MetaEpiAge"
path_work <- path_data
setwd(path_work)

###############################################
# Import and filtration
###############################################
myLoad <- champ.load(
  directory = path_data,
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
# Normalization and CpGs selection
###############################################
betas <- getBeta(preprocessFunnorm(myLoad$rgSet))
cpgs_orgn <- rownames(betas)
cpgs_fltd <- rownames(myLoad$beta)

###############################################
# PC clocks
# You need to setup path to 3 files from original repository (https://github.com/MorganLevineLab/PC-Clocks):
# 1) run_calcPCClocks.R
# 2) run_calcPCClocks_Accel.R
# 3) CalcAllPCClocks.RData (very big file but it is nesessary)
# You also need to apply changes from this issue: https://github.com/MorganLevineLab/PC-Clocks/issues/10
###############################################
source(paste(path_pc_clocks, "run_calcPCClocks.R", sep = ""))
source(paste(path_pc_clocks, "run_calcPCClocks_Accel.R", sep = ""))
pheno <- data.frame(
  'Sex' = pd$Sex,
  'Age' = pd$Age,
  'Tissue' = pd$Tissue
)
pheno['Female'] <- 1
pheno$Age <- as.numeric(pheno$Age)
pheno[pheno$Sex == 'M', 'Female'] <- 0
rownames(pheno) <- rownames(pd)
pc_clocks <- calcPCClocks(
  path_to_PCClocks_directory = path_pc_clocks,
  datMeth = t(betas),
  datPheno = pheno,
  column_check = "skip"
)
pc_clocks <- calcPCClocks_Accel(pc_clocks)
pc_ages <- list("PCHorvath1", "PCHorvath2", "PCHannum", "PCHannum", "PCPhenoAge", "PCGrimAge")
for (pc_age in pc_ages) {
  pd[rownames(pd), pc_age] <- pc_clocks[rownames(pd), pc_age]
}

###############################################
# Create data for Horvath's calculator
###############################################
cpgs_horvath_old <- read.csv(
  paste(path_horvath, "/cpgs_horvath_calculator.csv", sep=""),
  header=TRUE
)$CpG
cpgs_horvath_new <- read.csv(
  paste(path_horvath, "/datMiniAnnotation4_fixed.csv", sep=""),
  header=TRUE
)$Name
cpgs_horvath <- intersect(cpgs_horvath_old, rownames(betas))
cpgs_missed <- setdiff(cpgs_horvath_old, rownames(betas))
betas_missed <- matrix(data='NA', nrow=length(cpgs_missed), dim(betas)[2])
rownames(betas_missed) <- cpgs_missed
colnames(betas_missed) <- colnames(betas)
betas_horvath <- rbind(betas[cpgs_horvath, ], betas_missed)
betas_horvath <- data.frame(row.names(betas_horvath), betas_horvath)
colnames(betas_horvath)[1] <- "ProbeID"
write.csv(
  betas_horvath,
  file="betas_horvath.csv",
  row.names=FALSE,
  quote=FALSE
)

pheno_horvath <- data.frame(
  'Sex' = pd$Sex,
  'Age' = pd$Age,
  'Tissue' = pd$Tissue
)
pheno_horvath['Female'] <- 1
pheno_horvath[pheno_horvath$Sex == 'M', 'Female'] <- 0
pheno_horvath$Age <- as.numeric(pheno_horvath$Age)
rownames(pheno_horvath) <- rownames(pd)
pheno_horvath <- data.frame(row.names(pheno_horvath), pheno_horvath[ ,!(names(pheno_horvath) %in% c("Sex"))])
colnames(pheno_horvath)[1] <- "Sample_Name"
write.csv(
  pheno_horvath,
  file="pheno_horvath.csv",
  row.names=FALSE,
  quote=FALSE
)

###############################################
# DunedinPACE
###############################################
pace <- PACEProjector(betas)
pd['DunedinPACE'] <- pace$DunedinPACE

###############################################
# Save modified pheno
###############################################
write.csv(pd, file = "pheno11.csv")

###############################################
# Save DNAm data
###############################################
write.csv(betas, file = "betas.csv")
