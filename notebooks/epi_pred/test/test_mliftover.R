rm(list=ls())

###############################################
# Installing packages
###############################################
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
install.packages("devtools")
devtools::install_url("https://cran.r-project.org/src/contrib/Archive/kpmt/kpmt_0.1.0.tar.gz")
BiocManager::install("ChAMP")
library("ChAMP")
library("sesame")

###############################################
# Setting variables
###############################################
dataset_450 <- 'GSE87571'
arraytype_450 <- '450K'

dataset_EPIC <- 'GSE234461'
arraytype_EPIC <- 'EPIC'

###############################################
# Setting path
# Directory must contain *.idat files and *.csv file with phenotype
###############################################
path_data_450 <- "E:/YandexDisk/Work/bbd/epi_pred/GPL13534/24_GSE87571/raw"
path_data_EPIC <- "E:/YandexDisk/Work/bbd/epi_pred/GPL21145/93_GSE234461/raw"

###############################################
# Load common CpGs
###############################################
cpgs_intxn <- read.csv("E:/YandexDisk/Work/bbd/epi_pred/manifests/Illumina/intersection.csv")
rownames(cpgs_intxn) <- cpgs_intxn[,1]



###############################################
# Import and filtration
###############################################
path_work <- path_data_450
setwd(path_work)

myLoad_450 <- champ.load(
  directory = path_data_450,
  arraytype = arraytype_450,
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
pd_450 <- as.data.frame(myLoad_450$pd)

###############################################
# Functional normalization
###############################################
betas_450 <- getBeta(preprocessFunnorm(myLoad_450$rgSet))
write.csv(betas_450, file = "betas_450k.csv")

###############################################
# Import and filtration
###############################################
path_work <- path_data_EPIC
setwd(path_work)

myLoad_EPIC <- champ.load(
  directory = path_data_EPIC,
  arraytype = arraytype_EPIC,
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
pd_EPIC <-  as.data.frame(myLoad_EPIC$pd)

###############################################
# Functional normalization
###############################################
betas_EPIC <- getBeta(preprocessFunnorm(myLoad_EPIC$rgSet))
write.csv(betas_EPIC, file = "betas_EPIC.csv")

###############################################
# mLiftOver
###############################################
path_work <- path_data_450
setwd(path_work)

# Read file if it was saved before
betas_450 <- read.csv("E:/YandexDisk/Work/bbd/epi_pred/GPL13534/24_GSE87571/betas_450k.csv")
rownames(betas_450) <- betas_450[,1]
betas_450[,1] <- NULL
betas_450 <- data.matrix(betas_450, rownames.force = NA)


cpgs_cmn <- intersect(rownames(cpgs_intxn), rownames(betas_450))
betas_cmn <- betas_450[cpgs_cmn, ]
betas_cmn <- data.matrix(betas_cmn, rownames.force = NA)

betas_cmn_to_msa <- mLiftOver(data.matrix(betas_cmn, rownames.force = NA), "MSA", impute=FALSE)

betas_cmn_to_msa <- mLiftOver(data.matrix(betas_450, rownames.force = NA), "MSA", impute=FALSE)





betas_EPIC <- read.csv("E:/YandexDisk/DNAm draft/GEO/GPL21145/93_GSE234461/betas_EPIC.csv")
rownames(betas_EPIC) <- betas_EPIC[,1]
betas_EPIC[,1] <- NULL
betas_EPIC <- data.matrix(betas_EPIC, rownames.force = NA)

betas_450k_to_epic <- mLiftOver(betas_450, "EPIC", impute=FALSE)
betas_450k_to_epic_filtered <- betas_450k_to_epic[rowSums(!is.na(betas_450k_to_epic))>0,]
max(betas_450k_to_epic_filtered)
min(betas_450k_to_epic_filtered)
write.csv(betas_450k_to_epic, file = "betas_450k_to_epic.csv")

betas_450k_to_msa <- mLiftOver(betas_450, "MSA", impute=FALSE) # MSA has no option to be imputed, see sesameDataList()
betas_450k_to_msa_filtered <- betas_450k_to_msa[rowSums(!is.na(betas_450k_to_msa))>0,]
max(betas_450k_to_msa_filtered)
min(betas_450k_to_msa_filtered)
write.csv(betas_450k_to_msa, file = "betas_450k_to_msa.csv")

path_work <- path_data_EPIC
setwd(path_work)

betas_EPIC_to_msa <- mLiftOver(betas_EPIC, "MSA", impute=FALSE)
betas_EPIC_to_msa_filtered <- betas_EPIC_to_msa[rowSums(!is.na(betas_EPIC_to_msa))>0,]
max(betas_EPIC_to_msa_filtered)
min(betas_EPIC_to_msa_filtered)
write.csv(betas_EPIC_to_msa, file = "betas_EPIC_to_msa.csv")

betas_EPIC_to_450 <- mLiftOver(betas_EPIC, "HM450", impute=FALSE)
betas_EPIC_to_450_filtered <- betas_EPIC_to_450[rowSums(!is.na(betas_EPIC_to_450))>0,]
max(betas_EPIC_to_450_filtered)
min(betas_EPIC_to_450_filtered)
write.csv(betas_EPIC_to_450, file = "betas_EPIC_to_450.csv")

betas_EPIC_to_450_imputed <- mLiftOver(betas_EPIC, "HM450", impute=TRUE)
betas_EPIC_to_450_imputed_filtered <- betas_EPIC_to_450_imputed[rowSums(!is.na(betas_EPIC_to_450_imputed))>0,]
max(betas_EPIC_to_450_imputed_filtered)
min(betas_EPIC_to_450_imputed_filtered)
write.csv(betas_EPIC_to_450_imputed, file = "betas_EPIC_to_450_imputed.csv")
