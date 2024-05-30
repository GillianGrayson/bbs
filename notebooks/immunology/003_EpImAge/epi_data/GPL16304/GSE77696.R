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
dataset <- 'GSE77696'

###############################################
# Setting path
###############################################
path_data <- "D:/YandexDisk/Work/pydnameth/datasets/GPL16304/GSE77696"
path_pc_clocks <- "D:/YandexDisk/Work/pydnameth/datasets/lists/cpgs/PC_clocks/"
path_horvath <- "D:/YandexDisk/Work/pydnameth/draft/10_MetaEPIClock/MetaEpiAge"
path_work <- "D:/YandexDisk/Work/pydnameth/datasets/GPL21145/GSEUNN/special/060_EpiSImAge/GSE77696"
setwd(path_work)

###############################################
# Load annotations
###############################################
ann450k <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)

###############################################
# Import data
###############################################
pd <- as.data.frame(read_excel(paste(path_data,"/pheno.xlsx", sep="")))
sentrixids <- cSplit(pd, "description", ": ")
sentrixids$description_2 <- gsub(" ",".", as.character(sentrixids$description_2))
pd$sentrixids <- sentrixids$description_2
row.names(pd) <- pd$sentrixids
betas <- as.data.frame(read.table(paste(path_data,"/raw/GSE77696_MatrixProcessed.txt", sep=""), header=TRUE, sep='\t'))
rownames(betas) <- betas$ID_REF
col_odd <- seq_len(ncol(betas)) %% 2
betas <- betas[ , col_odd == 0]
missed_in_betas <- setdiff(row.names(pd), colnames(betas))
missed_in_pheno <- setdiff(colnames(betas), row.names(pd))
betas <- betas[, row.names(pd)]

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
write.csv(pd, file = "pheno.csv")

###############################################
# Save DNAm data
###############################################
cpgs_common <- intersect(rownames(betas), rownames(ann450k))
betas <- betas[cpgs_common, ]
write.csv(betas, file = "betas.csv")