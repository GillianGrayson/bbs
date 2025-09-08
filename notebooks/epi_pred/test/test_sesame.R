rm(list=ls())

if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
install.packages("remotes")
BiocManager::install("zwdzwd/sesame")

library(sesame)

dataset <- 'GSE87571'

path_data <- "E:/YandexDisk/DNAm draft/GEO/GPL13534/24_GSE87571/raw_short"
setwd(path_data)

betas <- openSesame(path_data, prep="QCDPB", collapseToPfx = TRUE)
betas_full <- do.call(cbind, lapply(searchIDATprefixes(path_data), function(pfx) {getBetas(noob(pOOBAH(dyeBiasNL(inferInfiniumIChannel(qualityMask(readIDATpair(pfx)))))))}))
betas_non_masked <- do.call(cbind, lapply(searchIDATprefixes(path_data), function(pfx) {getBetas(noob(pOOBAH(dyeBiasNL(inferInfiniumIChannel(qualityMask(readIDATpair(pfx)))))), mask=FALSE)}))

write.csv(betas, file = "betas.csv")

# Check how many CpGs were NaN
rows_with_all_nan <- apply(betas, 1, function(x) all(is.nan(x)))
rows_with_all_nan_subset <- betas[rows_with_all_nan, ]

rows_with_all_nan_full <- apply(betas_full, 1, function(x) all(is.na(x)))
rows_with_all_nan_subset_full <- betas_full[rows_with_all_nan_full, ]

rows_with_all_nan_non_masked <- apply(betas_non_masked, 1, function(x) all(is.na(x)))
rows_with_all_nan_subset_non_masked <- betas_non_masked[rows_with_all_nan_non_masked, ]

# Select only non-NaN CpGs
betas_filtered <- betas[rowSums(!is.nan(betas))>0,]

betas_filtered_full <- betas_full[rowSums(!is.na(betas_full))>0,]

betas_filtered_non_masked <- betas_non_masked[rowSums(!is.na(betas_non_masked))>0,]

# Check number of dropped CpGs on each step
betas_step_1 <- do.call(cbind, lapply(searchIDATprefixes(path_data), function(pfx) {getBetas(qualityMask(readIDATpair(pfx)))}))
rows_with_all_nan_step_1 <- apply(betas_step_1, 1, function(x) all(is.na(x)))
rows_with_all_nan_subset_step_1 <- betas_step_1[rows_with_all_nan_step_1, ]
betas_filtered_step_1 <- betas_step_1[rowSums(!is.na(betas_step_1))>0,]

betas_step_2 <- do.call(cbind, lapply(searchIDATprefixes(path_data), function(pfx) {getBetas(inferInfiniumIChannel(readIDATpair(pfx)))}))
rows_with_all_nan_step_2 <- apply(betas_step_2, 1, function(x) all(is.na(x)))
rows_with_all_nan_subset_step_2 <- betas_step_2[rows_with_all_nan_step_2, ]
betas_filtered_step_2 <- betas_step_2[rowSums(!is.na(betas_step_2))>0,]

betas_step_3 <- do.call(cbind, lapply(searchIDATprefixes(path_data), function(pfx) {getBetas(dyeBiasNL(readIDATpair(pfx)))}))
rows_with_all_nan_step_3 <- apply(betas_step_3, 1, function(x) all(is.na(x)))
rows_with_all_nan_subset_step_3 <- betas_step_3[rows_with_all_nan_step_3, ]
betas_filtered_step_3 <- betas_step_3[rowSums(!is.na(betas_step_3))>0,]

betas_step_4 <- do.call(cbind, lapply(searchIDATprefixes(path_data), function(pfx) {getBetas(pOOBAH(readIDATpair(pfx)))}))
rows_with_all_nan_step_4 <- apply(betas_step_4, 1, function(x) all(is.na(x)))
rows_with_all_nan_subset_step_4 <- betas_step_4[rows_with_all_nan_step_4, ]
betas_filtered_step_4 <- betas_step_4[rowSums(!is.na(betas_step_4))>0,]

betas_step_5 <- do.call(cbind, lapply(searchIDATprefixes(path_data), function(pfx) {getBetas(noob(readIDATpair(pfx)))}))
rows_with_all_nan_step_5 <- apply(betas_step_5, 1, function(x) all(is.na(x)))
rows_with_all_nan_subset_step_5 <- betas_step_5[rows_with_all_nan_step_5, ]
betas_filtered_step_5 <- betas_step_5[rowSums(!is.na(betas_step_5))>0,]

betas_step_6 <- do.call(cbind, lapply(searchIDATprefixes(path_data), function(pfx) {getBetas(matchDesign(readIDATpair(pfx)))}))
rows_with_all_nan_step_6 <- apply(betas_step_6, 1, function(x) all(is.na(x)))
rows_with_all_nan_subset_step_6 <- betas_step_6[rows_with_all_nan_step_6, ]
betas_filtered_step_6 <- betas_step_6[rowSums(!is.na(betas_step_6))>0,]
