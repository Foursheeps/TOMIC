#GSE123902
rm(list=ls())
library(dplyr)
library(Seurat)
df <- read.csv("../GSE123902_1200/features.csv", row.names = 1)

df_met <- df[which(df$domain=='source'),]
df_met$labels <- factor(df_met$labels)
df_met$pred   <- factor(df_met$pred, levels = levels(df_met$labels))

accuracy <- mean(df_met$labels == df_met$pred)
#Source domain ACC: 0.9727918

df_pt <- df[which(df$domain=='target'),]

library(readr)
library(purrr)

data_dir <- "../Data/GSE123902"

files <- c(
  "GSM3516662_MSK_LX653_PRIMARY_TUMOUR_dense.csv",
  "GSM3516663_MSK_LX661_PRIMARY_TUMOUR_dense.csv",
  "GSM3516665_MSK_LX675_PRIMARY_TUMOUR_dense.csv",
  "GSM3516667_MSK_LX676_PRIMARY_TUMOUR_dense.csv",
  "GSM3516669_MSK_LX679_PRIMARY_TUMOUR_dense.csv",
  "GSM3516670_MSK_LX680_PRIMARY_TUMOUR_dense.csv",
  "GSM3516672_MSK_LX682_PRIMARY_TUMOUR_dense.csv",
  "GSM3516672_MSK_LX682_PRIMARY_TUMOUR_dense.csv"
)

expr_list <- map(files, function(f) {
  df <- read.csv(file.path(data_dir, f), row.names = 1)
  prefix <- sub("_dense\\.csv$", "", f)
  rownames(df) <- paste0(prefix, "_", rownames(df))
  df
})

common_genes <- Reduce(intersect, lapply(expr_list, colnames))
length(common_genes)

expr_list <- lapply(expr_list, function(df) {
  df[, common_genes, drop = FALSE]
})

expr_all <- do.call(rbind, expr_list)
dim(expr_all)

cells_use <- intersect(rownames(df_pt), rownames(expr_all))
expr_sub <- expr_all[cells_use, , drop = FALSE]
dim(expr_sub)

#DEG analysis
stopifnot(all(rownames(expr_sub) %in% rownames(df_pt)))
df_pt <- df_pt[rownames(expr_sub), ]

expr_seu <- t(expr_sub)
seu <- CreateSeuratObject(counts = expr_seu)
seu$pred <- df_pt$pred
seu$pred <- factor(seu$pred)
Idents(seu) <- "pred"

deg_list <- list()

for (lab in levels(seu$pred)) {
  deg <- FindMarkers(
    object = seu,
    ident.1 = lab,
    ident.2 = setdiff(levels(seu$pred), lab),
    test.use = "wilcox",     # 或 "t", "LR", "MAST"
    logfc.threshold = 0.25,
    min.pct = 0.1
  )
  
  deg$gene <- rownames(deg)
  deg$class <- lab
  
  deg_list[[lab]] <- deg
}

# save
outdir <- "../Output/GSE123902_PT_DEG_results"
dir.create(outdir, showWarnings = FALSE) 
for (cls in names(deg_list)) {
  fname <- file.path(outdir, paste0(cls, "_DEG.csv"))
  write.csv(deg_list[[cls]], fname, row.names = TRUE)
}


deg <- deg_list[[1]]

deg <- deg %>%
  mutate(
    negLog10P = -log10(p_val_adj + 1e-300),  
    significance = case_when(
      p_val_adj < 0.05 & avg_log2FC > 0.5  ~ "Up",
      p_val_adj < 0.05 & avg_log2FC < -0.5 ~ "Down",
      TRUE ~ "NS"
    )
  )

num_up <- deg %>%
  filter(significance == "Up") %>%
  nrow()
num_up

# Volcano Plot
p1<-ggplot(deg, aes(x = avg_log2FC, y = negLog10P, color = significance)) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_color_manual(values = c("Up" = "red", "Down" = "blue", "NS" = "grey")) +
  theme_minimal() +
  xlab("log2 Fold Change") +
  ylab("-log10 Adjusted P-value") +
  ggtitle("XXX-specific MICs identified DEGs") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +
  theme(
    plot.title = element_text(hjust = 0.5),
    #legend.title = element_blank()
    legend.position = "none" 
  )



#################
#GSE163558
rm(list=ls())
df <- read.csv("../GSE163558_1200/features.csv", row.names = 1)

df_met <- df[which(df$domain=='source'),]
df_met$labels <- factor(df_met$labels)
df_met$pred   <- factor(df_met$pred, levels = levels(df_met$labels))

accuracy <- mean(df_met$labels == df_met$pred)
#Source domain ACC: .9242471

df_pt <- df[which(df$domain=='target'),]

pt1_data_dir <- "../Data/GSE163558/GSM5004180_PT1"
df_pt1 <- Read10X(data.dir = pt1_data_dir)
colnames(df_pt1) <- paste0('GSM5004180_PT1_',colnames(df_pt1))
dim(df_pt1)

pt2_data_dir <- "../Data/GSE163558/GSM5004181_PT2"
df_pt2 <- Read10X(data.dir = pt2_data_dir)
colnames(df_pt2) <- paste0('GSM5004181_PT2_',colnames(df_pt2))
dim(df_pt2)

pt3_data_dir <- "../Data/GSE163558/GSM5004182_PT3"
df_pt3 <- Read10X(data.dir = pt3_data_dir)
colnames(df_pt3) <- paste0('GSM5004182_PT3_',colnames(df_pt3))
dim(df_pt3)

expr_all <- cbind(df_pt1, df_pt2, df_pt3)
dim(expr_all)

#DEG analysis
df_pt <- df_pt[colnames(expr_all), ]

expr_seu <- expr_all
seu <- CreateSeuratObject(counts = expr_seu)
seu$pred <- df_pt$pred
seu$pred <- factor(seu$pred)
Idents(seu) <- "pred"

deg_list <- list()

for (lab in levels(seu$pred)) {
  deg <- FindMarkers(
    object = seu,
    ident.1 = lab,
    ident.2 = setdiff(levels(seu$pred), lab),
    test.use = "wilcox",     # 或 "t", "LR", "MAST"
    logfc.threshold = 0.25,
    min.pct = 0.1
  )
  
  deg$gene <- rownames(deg)
  deg$class <- lab
  
  deg_list[[lab]] <- deg
}

# Save
outdir <- "../Output/GSE163558_PT_DEG_results"
dir.create(outdir, showWarnings = FALSE) 
for (cls in names(deg_list)) {
  fname <- file.path(outdir, paste0(cls, "_DEG.csv"))
  write.csv(deg_list[[cls]], fname, row.names = TRUE)
}


library(ggplot2)
library(dplyr)

deg <- deg_list[[3]]

deg <- deg %>%
  mutate(
    negLog10P = -log10(p_val_adj + 1e-300), 
    significance = case_when(
      p_val_adj < 0.05 & avg_log2FC > 1  ~ "Up",
      p_val_adj < 0.05 & avg_log2FC < -1 ~ "Down",
      TRUE ~ "NS"
    )
  )

num_up <- deg %>%
  filter(significance == "Up") %>%
  nrow()
num_up

ggplot(deg, aes(x = avg_log2FC, y = negLog10P, color = significance)) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_color_manual(values = c("Up" = "red", "Down" = "blue", "NS" = "grey")) +
  theme_minimal() +
  xlab("log2 Fold Change") +
  ylab("-log10 Adjusted P-value") +
  ggtitle("XXX-specific MICs identified DEGs") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.title = element_blank()
  )
