rm(list=ls())

if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("DSS")
BiocManager::install("minfi")
BiocManager::install("ChAMPdata")
BiocManager::install("wateRmelon")
BiocManager::install("ChAMP")
BiocManager::install("methylGSA")
BiocManager::install("IlluminaHumanMethylationEPICv2anno.20a1.hg38")
install.packages("reticulate")
install.packages("readxl")
install.packages("stringr")

library(devtools)
devtools::install_github("YuanTian1991/ChAMP")
devtools::install_github("YuanTian1991/ChAMPData")
devtools::install_github("perishky/meffil")
devtools::install_github("ytwangZero/easyEWAS")

library("ChAMP")
library("methylGSA")
library(IlluminaHumanMethylationEPICv2anno.20a1.hg38)
library(readxl)
library(stringr)

####################################################################
### Python test
####################################################################
install.packages("reticulate")
Sys.setenv(RETICULATE_PYTHON = "C:/Users/alena/anaconda3/envs/py312/python.exe")
library("reticulate")
py_config()
Sys.which('python')
use_condaenv('py312')
py_run_string('print(1+1)')
rm(list=ls())
pd <- import("pandas")
path_load <- "E:/YandexDisk/pydnameth/datasets/GPL21145/GSEUNN/special/043_yakutia_EWAS/00_all_region/data_for_R"
pheno_py <- pd$read_pickle(paste(path_load, "/pheno_R_all_region.pkl", sep=''))
pheno_py$Region <- as.factor(pheno_py$Region)
betas_py <- pd$read_pickle(paste(path_load, "/betas_R_all_region.pkl", sep=''))
####################################################################

dmp_pval <- 1
dmr_pval <- 0.05
dmr_min_probes <- 10
gsea_pval <- 0.05
methylglm_minsize <- 10
methylglm_maxsize <- 1000

path <- "E:/YandexDisk/bbd/fmba/dnam/processed/special_63/funnorm"
setwd(path)

pheno <- read_excel("pheno_funnorm.xlsx")
pheno <- as.data.frame(pheno)
names(pheno) <- str_replace_all(names(pheno), c(" " = ".", "," = ""))
pheno$Special.Status <- as.factor(pheno$Special.Status)
colnames(pheno)[colnames(pheno) == '...1'] <- 'ID'
rownames(pheno) <- pheno[,1]
pheno <- pheno[,c("Age","Sex","Special.Status")]

betas <- read.csv("betas_funnorm.csv")
rownames(betas) <- betas[,1]
betas[,1] <- NULL
colnames(betas) <- gsub("^X", "", colnames(betas))
####################################################################
### DMP function test
####################################################################
dmp <- champ.DMP(
  beta = betas,
  pheno = pheno$Special.Status,
  compare.group = c("Control", "Case"),
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPICv2"
)
dmp_short <- dmp$Control_to_Case[dmp$Control_to_Case$adj.P.Val<=0.05,]
if (!all(is.na(dmp_short))) {
  write.csv(dmp_short, file = "DMP_orgn_champ.csv")
}

cpgs_fltr <- read.csv("cpgs_fltd.csv")
cpgs_fltr <- as.character(cpgs_fltr[,1])
betas_fltr <- betas[row.names(betas) %in% cpgs_fltr,]
dmp_fltr <- champ.DMP(
  beta = betas_fltr,
  pheno = pheno$Special.Status,
  compare.group = c("Control", "Case"),
  adjPVal = dmp_pval,
  adjust.method = "BH",
  arraytype = "EPICv2"
)
dmp_short_fltr <- dmp_fltr$Control_to_Case[dmp_fltr$Control_to_Case$adj.P.Val<=0.05,]
if (!all(is.na(dmp_short_fltr))) {
  write.csv(dmp_short_fltr, file = "DMP_fltr_champ.csv")
}
####################################################################
### DMR function test
####################################################################
dmr <- champ.DMR(
  beta = data.matrix(betas),
  pheno = pheno$Special.Status,
  compare.group = c("Control", "Case"),
  arraytype = "EPICv2",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 8,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr$BumphunterDMR, file = "DMR_orgn_champ.csv")

dmr_fltr <- champ.DMR(
  beta = data.matrix(betas_fltr),
  pheno = pheno$Special.Status,
  compare.group = c("Control", "Case"),
  arraytype = "EPICv2",
  method = "Bumphunter", # "Bumphunter" "ProbeLasso" "DMRcate"
  minProbes = dmr_min_probes,
  adjPvalDmr = dmr_pval,
  cores = 8,
  ## following parameters are specifically for Bumphunter method.
  maxGap = 300,
  cutoff = NULL,
  pickCutoff = TRUE,
  smooth = TRUE,
  smoothFunction = loessByCluster,
  useWeights = FALSE,
  permutations = NULL,
  B = 250,
  nullMethod = "bootstrap"
)
write.csv(dmr_fltr$BumphunterDMR, file = "DMR_fltr_champ.csv")

RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPICv2", annotation = "20a1.hg38"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_genes_orgn_champ.csv", row.names=FALSE)

RSobject <- RatioSet(betas_fltr, annotation = c(array = "IlluminaHumanMethylationEPICv2", annotation = "20a1.hg38"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(apply(dmr[[1]], 1, function(x) rownames(RSanno)[which(RSanno$chr == x[1] & RSanno$pos >= as.numeric(x[2]) & RSanno$pos <= as.numeric(x[3]))])))
loi.lv[["DMR"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$DMR), file = "DMR_genes_fltr_champ.csv", row.names=FALSE)
####################################################################
### GSEA function test for DMP/DMR
####################################################################
gsea <- champ.GSEA(beta=betas, 
  DMP=dmp[[1]],
  DMR=dmr,
  CpGlist=NULL,
  Genelist=NULL,
  pheno=pheno$Special.Status,
  method="fisher",
  arraytype="EPICv2",
  Rplot=TRUE,
  adjPval=gsea_pval,
  cores=8)
if (!all(is.na(gsea$DMP))) {
  gtResult_DMP_orgn <- data.frame(row.names(gsea$DMP), gsea$DMP)
  write.csv(gtResult_DMP_orgn, file = "GSEA(fisher)_DMP_orgn_champ.csv", row.names=TRUE)
}
if (!all(is.na(gsea$DMR))) {
  gtResult_DMR_orgn <- data.frame(row.names(gsea$DMR), gsea$DMR)
  write.csv(gtResult_DMR_orgn, file = "GSEA(fisher)_DMR_orgn_champ.csv", row.names=TRUE)
}

gsea_fltr <- champ.GSEA(beta=betas_fltr, 
  DMP=dmp_fltr[[1]],
  DMR=dmr_fltr,
  CpGlist=NULL,
  Genelist=NULL,
  pheno=pheno$Special.Status,
  method="fisher",
  arraytype="EPICv2",
  Rplot=TRUE,
  adjPval=gsea_pval,
  cores=8)
if (!all(is.na(gsea_fltr$DMP))) {
  gtResult_DMP_fltr <- data.frame(row.names(gsea_fltr$DMP), gsea_fltr$DMP)
  write.csv(gtResult_DMP_fltr, file = "GSEA(fisher)_DMP_fltr_champ.csv", row.names=TRUE)
}
if (!all(is.na(gsea_fltr$DMR))) {
  gtResult_DMR_fltr <- data.frame(row.names(gsea_fltr$DMR), gsea_fltr$DMR)
  write.csv(gtResult_DMR_fltr, file = "GSEA(fisher)_DMR_fltr_champ.csv", row.names=TRUE)
}
####################################################################
### Seems that GSEA function does not work with EPICv2
####################################################################
gsea <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = NULL,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Special.Status,
  method = "ebayes",
  arraytype = "EPICv2",
  Rplot = FALSE,
  adjPval = gsea_pval,
  cores = 8
)
gtResult <- data.frame(row.names(gsea[[3]]), gsea[[3]])
colnames(gtResult)[1] <- "ID"
write.csv(gtResult, file = "GSEA(ebayes)_gtResult_orgn.csv", row.names=TRUE)
write.csv(gsea$GSEA[[1]], file = "GSEA(ebayes)_Rank(P)_orgn.csv", row.names=TRUE)
####################################################################
### Seems that methylglm function does not work with EPICv2
####################################################################
dmp_df <- data.frame(row.names(dmp$Control_to_Case), dmp$Control_to_Case)
colnames(dmp_df)[1] <- "CpG"
cpg_pval <- setNames(dmp_df$adj.P.Val, dmp_df$CpG)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPICv2",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "GO",
  minsize = methylglm_minsize,
  maxsize = methylglm_maxsize,
  parallel = TRUE
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_GO_orgn.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "KEGG",
  minsize = methylglm_minsize,
  maxsize = methylglm_maxsize,
  parallel = TRUE
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_KEGG_orgn.csv", row.names=FALSE)
GSEA_methylglm <- methylglm(
  cpg.pval = cpg_pval,
  array.type = "EPIC",
  group = "all",
  GS.idtype = "SYMBOL",
  GS.type = "Reactome",
  minsize = methylglm_minsize,
  maxsize = methylglm_maxsize,
  parallel = TRUE
)
write.csv(GSEA_methylglm, file = "GSEA(methylglm)_Reactome_orgn.csv", row.names=FALSE)
####################################################################
### Seems that GSEA function with gometh method does not work with EPICv2
####################################################################
GSEA_gometh <- champ.GSEA(
  beta = betas,
  DMP = NULL,
  DMR = dmr,
  CpGlist = NULL,
  Genelist = NULL,
  pheno = pheno$Special.Status,
  method = "gometh",
  arraytype = "EPICv2",
  Rplot = TRUE,
  adjPval = dmr_pval,
  cores = 8
)
write.csv(data.frame(GSEA_gometh$DMR), file = "DMR_GSEA_gometh_orgn.csv", row.names=FALSE)
####################################################################

####################################################################
### limma testing
####################################################################
rm(list=ls())

library("ChAMP")
library("methylGSA")
library(IlluminaHumanMethylationEPICv2anno.20a1.hg38)
library(readxl)
library(stringr)
library(limma)
library(missMethyl)

path <- "E:/YandexDisk/bbd/fmba/dnam/processed/special_63/funnorm"
setwd(path)

pheno <- read_excel("pheno_funnorm.xlsx")
pheno <- as.data.frame(pheno)
names(pheno) <- str_replace_all(names(pheno), c(" " = ".", "," = ""))
pheno$Special.Status <- as.factor(pheno$Special.Status)
colnames(pheno)[colnames(pheno) == '...1'] <- 'ID'
rownames(pheno) <- pheno[,1]
pheno <- pheno[,c("Age","Sex","Special.Status")]

betas <- read.csv("betas_funnorm.csv")
rownames(betas) <- betas[,1]
betas[,1] <- NULL
colnames(betas) <- gsub("^X", "", colnames(betas))

cpgs_fltr <- read.csv("cpgs_fltd.csv")
cpgs_fltr <- as.character(cpgs_fltr[,1])
betas_fltr <- betas[row.names(betas) %in% cpgs_fltr,]

betas <- betas[grepl("^cg", rownames(betas)),]
betas_fltr <- betas_fltr[grepl("^cg", rownames(betas_fltr)),]

group <- factor(pheno$Special.Status, levels=c("Control","Case"))
age <- pheno$Age
design <- model.matrix(~group)
row.names(design) <- row.names(pheno)

fit.reduced <- lmFit(betas, design)
fit.reduced <- eBayes(fit.reduced, proportion=0.01, robust=TRUE)
top <- topTable(fit.reduced, adjust="BH", sort.by="B", number=nrow(fit.reduced))
write.csv(top, file = "GSEA(ebayes)_group_orgn_limma.csv", row.names=TRUE)

design_for_contrast <- model.matrix(~group+age)
row.names(design_for_contrast) <- row.names(pheno)
design_contrast <- makeContrasts(GroupWoAge=groupCase-age, levels=design_for_contrast)

fit.contrast <- lmFit(betas, design_for_contrast)
fit.reduced.contrast <- contrasts.fit(fit.contrast, design_contrast)
fit.reduced.contrast <- eBayes(fit.reduced.contrast, proportion=0.01, robust=TRUE)
top.contrast <- topTable(fit.reduced.contrast, adjust="BH", sort.by="B", number=nrow(fit.reduced.contrast))
write.csv(top.contrast, file = "GSEA(ebayes)_group_wo_age_orgn_limma.csv", row.names=TRUE)

top_short <- top[top$adj.P.Val<=0.05,]
if (!all(is.na(top_short))) {
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPICv2", annotation = "20a1.hg38"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(row.names(top_short)))
loi.lv[["CpG"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$CpG), file = "GSEA(ebayes)_group_genes_orgn_limma.csv", row.names=FALSE)
}

top_contrast_short <- top.contrast[top.contrast$adj.P.Val<=0.05,]
if (!all(is.na(top_contrast_short))) {
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPICv2", annotation = "20a1.hg38"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(row.names(top_contrast_short)))
loi.lv[["CpG"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$CpG), file = "GSEA(ebayes)_group_wo_age_genes_orgn_limma.csv", row.names=FALSE)
}

fit.reduced.fltr <- lmFit(betas_fltr, design)
fit.reduced.fltr  <- eBayes(fit.reduced.fltr , proportion=0.01, robust=TRUE)
top.fltr <- topTable(fit.reduced.fltr , adjust="BH", sort.by="B", number=nrow(fit.reduced.fltr ))
write.csv(top.fltr, file = "GSEA(ebayes)_group_fltr_limma.csv", row.names=TRUE)

fit.contrast.fltr <- lmFit(betas_fltr, design_for_contrast)
fit.reduced.contrast.fltr <- contrasts.fit(fit.contrast.fltr, design_contrast)
fit.reduced.contrast.fltr <- eBayes(fit.reduced.contrast.fltr, proportion=0.01, robust=TRUE)
top.contrast.fltr <- topTable(fit.reduced.contrast.fltr, adjust="BH", sort.by="B", number=nrow(fit.reduced.contrast.fltr))
write.csv(top.contrast.fltr, file = "GSEA(ebayes)_group_wo_age_fltr_limma.csv", row.names=TRUE)

top_short_fltr <- top.fltr[top.fltr$adj.P.Val<=0.05,]
if (!all(is.na(top_short_fltr))) {
RSobject <- RatioSet(betas_fltr, annotation = c(array = "IlluminaHumanMethylationEPICv2", annotation = "20a1.hg38"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(row.names(top_short)))
loi.lv[["CpG"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$CpG), file = "GSEA(ebayes)_group_genes_fltr_limma.csv", row.names=FALSE)
}

top_contrast_short_fltr <- top.contrast.fltr[top.contrast.fltr$adj.P.Val<=0.05,]
if (!all(is.na(top_contrast_short_fltr))) {
RSobject <- RatioSet(betas_fltr, annotation = c(array = "IlluminaHumanMethylationEPICv2", annotation = "20a1.hg38"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(row.names(top_contrast_short)))
loi.lv[["CpG"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$CpG), file = "GSEA(ebayes)_group_wo_age_genes_fltr_limma.csv", row.names=FALSE)
}

### GSEA gometh
cpgs_orgn <- read.csv("cpgs_orgn.csv")
cpgs_orgn <- as.character(cpgs_orgn[,1])

gsea_res_all <- gometh(
  row.names(top_short),
  all.cpg = cpgs_orgn,
  collection = c("GO", "KEGG"),
  array.type = "EPIC_V2")

gsea_res_adj <- gsea_res_all[gsea_res_all$FDR<=0.05,]
if (!all(is.na(gsea_res_adj))) {
  write.csv(gsea_res_adj, file = "GSEA(gometh)_group_orgn.csv", row.names=TRUE)
}

### Plotting CpGs
cpgs <- rownames(top)
par(mfrow=c(2,2))
for(i in 1:4){
  selected_betas <- as.numeric(betas[row.names(betas)==cpgs[i],])
  stripchart(selected_betas~design[, 2],method="jitter",
  group.names=c("Control","Case"),pch=16,cex=1.5,col=c(4,2),ylab="Beta values",
  vertical=TRUE,cex.axis=1.5,cex.lab=1.5)
  title(cpgs[i],cex.main=1.5)
}

####################################################################
### GSEA testing for regions (DMRcate)
####################################################################
rm(list=ls())

library("ChAMP")
library("methylGSA")
library(IlluminaHumanMethylationEPICv2anno.20a1.hg38)
library(readxl)
library(stringr)
library(limma)
library(DMRcate)

path <- "E:/YandexDisk/bbd/fmba/dnam/processed/special_63/funnorm"
setwd(path)

pheno <- read_excel("pheno_funnorm.xlsx")
pheno <- as.data.frame(pheno)
names(pheno) <- str_replace_all(names(pheno), c(" " = ".", "," = ""))
pheno$Special.Status <- as.factor(pheno$Special.Status)
colnames(pheno)[colnames(pheno) == '...1'] <- 'ID'
rownames(pheno) <- pheno[,1]
pheno <- pheno[,c("Age","Sex","Special.Status")]

betas <- read.csv("betas_funnorm.csv")
rownames(betas) <- betas[,1]
betas[,1] <- NULL
colnames(betas) <- gsub("^X", "", colnames(betas))

cpgs_fltr <- read.csv("cpgs_fltd.csv")
cpgs_fltr <- as.character(cpgs_fltr[,1])
betas_fltr <- betas[row.names(betas) %in% cpgs_fltr,]

group <- factor(pheno$Special.Status, levels=c("Control","Case"))
age <- pheno$Age
design <- model.matrix(~group)
row.names(design) <- row.names(pheno)

design_for_contrast <- model.matrix(~group+age)
row.names(design_for_contrast) <- row.names(pheno)
design_contrast <- makeContrasts(GroupWoAge=groupCase-age, levels=design_for_contrast)

annotation_diff <- cpg.annotate(
  datatype="array", 
  object=data.matrix(betas), what="Beta", 
  arraytype="EPICv2", 
  analysis.type="differential",
  design=design, 
  contrasts=FALSE, cont.matrix=NULL, 
  fdr=0.05, coef=2)
diff_DMRs <- dmrcate(annotation_diff, lambda=1000, C=2)
results.ranges.diff <- extractRanges(diff_DMRs)
results.ranges.diff.sign <- results.ranges.diff[results.ranges.diff$HMFDR<=0.05,]
if (!all(is.na(results.ranges.diff.sign))) {
  results.ranges.diff.sign.df <- as.data.table(results.ranges.diff.sign)
  results.ranges.diff.sign.df2 = data.frame(lapply(results.ranges.diff.sign.df, as.character), stringsAsFactors=FALSE)
  write.csv(results.ranges.diff.sign.df2, file = "DMRcate_diff_group_orgn.csv", row.names=TRUE)
}
if (!all(is.na(results.ranges.diff.sign.df2))) {
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPICv2", annotation = "20a1.hg38"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(row.names(results.ranges.diff.sign.df2)))
loi.lv[["CpG"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$CpG), file = "DMRcate_group_genes_orgn.csv", row.names=FALSE)
}

annotation_contrast_diff <- cpg.annotate(
  datatype="array", 
  object=data.matrix(betas), what="Beta", 
  arraytype="EPICv2", 
  analysis.type="differential",
  design=design_for_contrast, 
  contrasts=TRUE, cont.matrix=design_contrast, 
  fdr=0.05, coef=colnames(design_contrast)[1])
diff_contrast_DMRs <- dmrcate(annotation_contrast_diff, lambda=1000, C=2)
if (diff_contrast_DMRs) {
  results.ranges.diff.contrast <- extractRanges(diff_contrast_DMRs)
  results.ranges.diff.contrast.sign <- results.ranges.diff.contrast[results.ranges.diff.contrast$HMFDR<=0.05,]
  if (!all(is.na(results.ranges.diff.contrast.sign))) {
    write.csv(results.ranges.diff.contrast.sign, file = "DMRcate_diff_contrast_group_orgn.csv", row.names=TRUE)
  }
}

annotation_diff_fltr <- cpg.annotate(
  datatype="array", 
  object=data.matrix(betas_fltr), what="Beta", 
  arraytype="EPICv2", 
  analysis.type="differential",
  design=design, 
  contrasts=FALSE, cont.matrix=NULL, 
  fdr=0.05, coef=2)
diff_DMRs_fltr <- dmrcate(annotation_diff_fltr, lambda=1000, C=2)
results.ranges.diff.fltr <- extractRanges(diff_DMRs_fltr)
results.ranges.diff.sign.fltr <- results.ranges.diff.fltr[results.ranges.diff.fltr$HMFDR<=0.05,]
if (!all(is.na(results.ranges.diff.sign.fltr))) {
  results.ranges.diff.sign.df.fltr <- as.data.table(results.ranges.diff.sign.fltr)
  results.ranges.diff.sign.df2.fltr = data.frame(lapply(results.ranges.diff.sign.df.fltr, as.character), stringsAsFactors=FALSE)
  write.csv(results.ranges.diff.sign.df2.fltr, file = "DMRcate_diff_group_fltr.csv", row.names=TRUE)
}
if (!all(is.na(results.ranges.diff.sign.df2.fltr))) {
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPICv2", annotation = "20a1.hg38"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(row.names(results.ranges.diff.sign.df2.fltr)))
loi.lv[["CpG"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$CpG), file = "DMRcate_group_genes_fltr.csv", row.names=FALSE)
}

annotation_contrast_diff_fltr <- cpg.annotate(
  datatype="array", 
  object=data.matrix(betas_fltr), what="Beta", 
  arraytype="EPICv2", 
  analysis.type="differential",
  design=design_for_contrast, 
  contrasts=TRUE, cont.matrix=design_contrast, 
  fdr=0.05, coef=colnames(design_contrast)[1])
diff_contrast_DMRs_fltr <- dmrcate(annotation_contrast_diff_fltr, lambda=1000, C=2)
if (diff_contrast_DMRs_fltr) {
  results.ranges.diff.contrast.fltr <- extractRanges(diff_contrast_DMRs_fltr)
  results.ranges.diff.contrast.sign.fltr <- results.ranges.diff.contrast.fltr[results.ranges.diff.contrast.fltr$HMFDR<=0.05,]
  if (!all(is.na(results.ranges.diff.contrast.sign.fltr))) {
    write.csv(results.ranges.diff.contrast.sign.fltr, file = "DMRcate_diff_contrast_group_fltr.csv", row.names=TRUE)
  }
}

### Visualization + GSEA

cols <- c(2,4)[pheno$Special.Status]
names(cols) <- pheno$Special.Status
par(mfrow=c(1,1))
DMR.plot(ranges=results.ranges.diff, dmr=2, CpGs=betas, phen.col=cols, 
         what="Beta", arraytype="EPICv2", genome="hg38")

gst.region <- goregion(results.ranges.diff, all.cpg=rownames(betas), 
                       collection="GO", array.type="EPICv2", plot.bias=TRUE)

gst.region.kegg <- goregion(results.ranges.diff, all.cpg=rownames(betas), 
                       collection="KEGG", array.type="EPICv2")

gsa.region <- gsaregion(results.ranges.diff, all.cpg=rownames(betas), 
                        collection=hallmark)

### Three next options (var, ANOVA, diffVar) don't work with EPICv2

annotation_var <- cpg.annotate(
  datatype="array", 
  object=na.omit(data.matrix(betas)), what="Beta", 
  arraytype="EPICv2", 
  analysis.type="variability", na.rm=TRUE)
var_DMRs <- dmrcate(annotation_var, lambda=1000, C=2)
if (var_DMRs) {
  results.ranges.var <- extractRanges(var_DMRs)
  results.ranges.var.sign <- results.ranges.var[results.ranges.var$HMFDR<=0.05,]
  if (!all(is.na(results.ranges.var.sign))) {
    write.csv(results.ranges.var.sign, file = "DMRcate_var_group_orgn.csv", row.names=TRUE)
  }
}

annotation_ANOVA <- cpg.annotate(
  datatype="array", 
  object=data.matrix(betas), what="Beta", 
  arraytype="EPICv2", 
  analysis.type="ANOVA", design=design, 
  fdr = 0.05) 
anova_DMRs <- dmrcate(annotation_ANOVA, lambda=1000, C=2)
if (diff_DMRs) {
  results.ranges.anova <- extractRanges(anova_DMRs)
  results.ranges.anova.sign <- results.ranges.anova[results.ranges.anova$HMFDR<=0.05,]
  if (!all(is.na(results.ranges.anova.sign))) {
    write.csv(results.ranges.anova.sign, file = "DMRcate_anova_group_orgn.csv", row.names=TRUE)
  }
}

annotation_diff_var <- cpg.annotate(
  datatype="array", 
  object=na.omit(data.matrix(betas)), what="Beta", 
  arraytype="EPICv2", 
  analysis.type="diffVar", design=design, 
  contrasts = FALSE, cont.matrix = NULL, 
  fdr=0.05, varFitcoef=2) 
diff_var_DMRs <- dmrcate(annotation_diff_var, lambda=1000, C=2)
if (diff_var_DMRs) {
  results.ranges.diff.var <- extractRanges(diff_var_DMRs)
  results.ranges.diff.var.sign <- results.ranges.diff.var[results.ranges.diff.var$HMFDR<=0.05,]
  if (!all(is.na(results.ranges.diff.var.sign))) {
    write.csv(results.ranges.diff.var.sign, file = "DMRcate_diff_var_group_orgn.csv", row.names=TRUE)
  }
}

####################################################################
### meffil testing
####################################################################
rm(list=ls())

library(meffil)
library(IlluminaHumanMethylationEPICv2anno.20a1.hg38)
library(readxl)
library(stringr)

path <- "E:/YandexDisk/bbd/fmba/dnam/processed/special_63/funnorm"
setwd(path)

pheno <- read_excel("pheno_funnorm.xlsx")
pheno <- as.data.frame(pheno)
names(pheno) <- str_replace_all(names(pheno), c(" " = ".", "," = ""))
pheno$Special.Status <- as.factor(pheno$Special.Status)
colnames(pheno)[colnames(pheno) == '...1'] <- 'ID'
rownames(pheno) <- pheno[,1]
pheno <- pheno[,c("Age","Sex","Special.Status")]

betas <- read.csv("betas_funnorm.csv")
rownames(betas) <- betas[,1]
betas[,1] <- NULL
colnames(betas) <- gsub("^X", "", colnames(betas))

group <- pheno$Special.Status
age <- data.frame(Age = pheno$Age)
rownames(age) <- rownames(pheno)

beta.nodup <- meffil.collapse.dups(data.matrix(betas))

set.seed(1337)  
ewas.ret <- meffil.ewas(beta.nodup, variable=pheno$Special.Status, covariates=NULL, isva=F) 

ewas.parameters <- meffil.ewas.parameters(sig.threshold=0.05,  ## EWAS p-value threshold
                                          max.plots=10, ## plot at most 10 CpG sites
                                          qq.inflation.method="median",  ## measure inflation using median
                                          model="sva") ## select default EWAS model; 

ewas.summary<-meffil.ewas.summary(ewas.ret,beta.nodup,parameters=ewas.parameters)                              

meffil.ewas.report(ewas.summary, output.file="meffil_ewas_report.html")

set.seed(1337)
ewas.ret.cont <- meffil.ewas(beta.nodup, variable=group, covariates=age, isva=F) 

ewas.parameters <- meffil.ewas.parameters(sig.threshold=0.05,  ## EWAS p-value threshold
                                          max.plots=10, ## plot at most 10 CpG sites
                                          qq.inflation.method="median",  ## measure inflation using median
                                          model="sva") ## select default EWAS model; 

ewas.summary.cont<-meffil.ewas.summary(ewas.ret.cont,beta.nodup,parameters=ewas.parameters)                              

meffil.ewas.report(ewas.summary.cont, output.file="meffil_cont_ewas_report.html")

####################################################################
### easyEWAS testing
####################################################################
rm(list=ls())

library(easyEWAS)
library(IlluminaHumanMethylationEPICv2anno.20a1.hg38)
library(readxl)
library(stringr)

path <- "E:/YandexDisk/bbd/fmba/dnam/processed/special_63/funnorm"
setwd(path)

pheno <- read_excel("pheno_funnorm.xlsx")
pheno <- as.data.frame(pheno)
names(pheno) <- str_replace_all(names(pheno), c(" " = ".", "," = ""))
pheno$Special.Status <- as.factor(pheno$Special.Status)
colnames(pheno)[colnames(pheno) == '...1'] <- 'ID'
rownames(pheno) <- pheno[,1]
rownames(pheno) <- as.character(rownames(pheno))
pheno[,1] <- as.character(pheno[,1])
pheno <- pheno[,c("ID", "Age","Sex","Special.Status")]

betas <- read.csv("betas_funnorm.csv")
rownames(betas) <- betas[,1]
colnames(betas) <- gsub("^X", "", colnames(betas))
colnames(betas) <- as.character(colnames(betas))

# prepare the data file ------
res <- initEWAS(outpath = path)
res <- loadEWAS(input = res,
                ExpoData = pheno,
                MethyData = betas)

res <- transEWAS(input = res, Vars = "Special.Status", TypeTo = "factor")
                 
# perform the EWAS analysis ------
res <- startEWAS(input = res,
                model = "lm",
                expo = "Special.Status",
                cov = "Age",
                core = "default")

# visualize the EWAS result ------
res <- plotEWAS(input = res,
                file = "jpg",
                p = "PVAL_1",
                threshold = 0.05)

# internal validation based on the bootstrap method ------
res <- bootEWAS(input = res,
                filterP = "PVAL_1",
                cutoff = 0.001,
                bootCI = "perc",
                times = 100)

# conduct enrichment analysis ------
res <- enrichEWAS(input = res,
                  method = "KEGG",
                  filterP = "PVAL_1",
                  cutoff = 0.05,
                  plot = TRUE,
                  plotType = "dot",
                  plotcolor = "pvalue",
                  showCategory = 20)

# DMR analysis -----
res <- dmrEWAS(input = res,
               chipType = "EPICV2",
               what = "Beta",
               expo = "Special.Status",
               cov = "Age",
               genome = "hg38",
               lambda=1000,
               C = 2,
               filename = "default",
               pcutoff = 0.05,
               epicv2Filter = "mean")

####################################################################
### minfi testing
####################################################################
rm(list=ls())

library(minfi)
library(IlluminaHumanMethylationEPICv2anno.20a1.hg38)
library(readxl)
library(stringr)

path <- "E:/YandexDisk/bbd/fmba/dnam/processed/special_63/funnorm"
setwd(path)

pheno <- read_excel("pheno_funnorm.xlsx")
pheno <- as.data.frame(pheno)
names(pheno) <- str_replace_all(names(pheno), c(" " = ".", "," = ""))
pheno$Special.Status <- as.factor(pheno$Special.Status)
colnames(pheno)[colnames(pheno) == '...1'] <- 'ID'
rownames(pheno) <- pheno[,1]
pheno <- pheno[,c("Age","Sex","Special.Status")]

betas <- read.csv("betas_funnorm.csv")
rownames(betas) <- betas[,1]
betas[,1] <- NULL
colnames(betas) <- gsub("^X", "", colnames(betas))

dmp <- dmpFinder(data.matrix(betas), pheno=pheno$Special.Status, type="continuous")
write.csv(data.frame(dmp), file = "DMP_group_orgn_minfi.csv", row.names=TRUE)

dmp_short <- dmp[dmp$qval<=0.05,]
if (!all(is.na(dmp_short))) {
RSobject <- RatioSet(betas, annotation = c(array = "IlluminaHumanMethylationEPICv2", annotation = "20a1.hg38"))
RSanno <- getAnnotation(RSobject)[, c("chr", "pos", "Name", "UCSC_RefGene_Name")]
loi.lv <- list()
cpg.idx <- unique(unlist(row.names(dmp_short)))
loi.lv[["CpG"]] <- unique(unlist(sapply(RSanno[cpg.idx, "UCSC_RefGene_Name"], function(x) strsplit(x, split = ";")[[1]])))
write.csv(data.frame(loi.lv$CpG), file = "DMP_group_genes_orgn_minfi.csv", row.names=FALSE)
}

group <- factor(pheno$Special.Status, levels=c("Control","Case"))
age <- pheno$Age
design <- model.matrix(~group)
row.names(design) <- row.names(pheno)

GRset <- makeGenomicRatioSetFromMatrix(data.matrix(betas), array="IlluminaHumanMethylationEPICv2", annotation="20a1.hg38", mergeManifest=TRUE, what="Beta")
dmr <- bumphunter(GRset, design, coef=2, type="Beta", cutoff=0.05)

if (!all(is.na(dmr$table))) {
  write.csv(dmr$table, file = "DMR_orgn_minfi.csv", row.names=TRUE)
}
