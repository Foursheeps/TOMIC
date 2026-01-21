rm(list=ls())
library(Seurat)
library(ggplot2)
library(dplyr)

exp_0h  <- Read10X("../Data/GSE249057/GSM7925719_0h")
exp_6h  <- Read10X("../Data/GSE249057/GSM7925720_6h")
exp_48h <- Read10X("../Data/GSE249057/GSM7925721_48h")
exp_2mo <- Read10X("../Data/GSE249057/GSM7925722_2mo")
exp_4mo <- Read10X("../Data/GSE249057/GSM7925723_4mo")

colnames(exp_0h)  <- paste0("0h_",  colnames(exp_0h))
colnames(exp_6h)  <- paste0("6h_",  colnames(exp_6h))
colnames(exp_48h) <- paste0("48h_", colnames(exp_48h))
colnames(exp_2mo) <- paste0("2mo_", colnames(exp_2mo))
colnames(exp_4mo) <- paste0("4mo_", colnames(exp_4mo))

exp_all <- cbind(exp_0h, exp_6h, exp_48h, exp_2mo, exp_4mo)
dim(exp_all)

seu <- CreateSeuratObject(
  counts = exp_all,
  min.cells = 3,
  min.features = 200,
  project = "TimeCourse"
)

seu$timepoint <- gsub("_.*", "", colnames(seu))
table(seu$timepoint)

seu[["percent.mt"]] <- PercentageFeatureSet(seu, pattern = "^MT-")

VlnPlot(
  seu,
  features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
  group.by = "timepoint",
  pt.size = 0
)

qc_before <- seu@meta.data %>%
  count(timepoint) %>%
  mutate(stage = "Before QC")
qc_before

seu_qc <- subset(
  seu,
  subset = nFeature_RNA > 500 &
    nFeature_RNA < 7500 &
    nCount_RNA > 1500 &
    percent.mt < 20
)

qc_after <- seu_qc@meta.data %>%
  count(timepoint) %>%
  mutate(stage = "After QC")
qc_after
#timepoint    n    stage
#1        0h 9042 After QC
#2       2mo 3501 After QC
#3       48h 1130 After QC
#4       4mo 4530 After QC
#5        6h 2026 After QC

seu_qc <- NormalizeData(seu_qc)
seu_qc <- FindVariableFeatures(seu_qc, nfeatures = 3000)
seu_qc <- ScaleData(
  seu_qc,
  vars.to.regress = c("nCount_RNA", "percent.mt")
)
seu_qc <- RunPCA(seu_qc, npcs = 50)
seu_qc <- RunUMAP(seu_qc, dims = 1:30)

DimPlot(
  seu_qc,
  reduction = "umap",
  group.by = "timepoint",
  pt.size = 0.4
)


#Seurat batch effect removal
seu.list <- SplitObject(seu_qc, split.by = "timepoint")

library(future)
options(future.globals.maxSize = 2000 * 1024^2)

#variable features
seu.list <- lapply(seu.list, function(x) {
  x <- SCTransform(x, verbose = FALSE)
})

features <- SelectIntegrationFeatures(
  object.list = seu.list, 
  nfeatures = 3000
)

seu.list <- PrepSCTIntegration(
  object.list = seu.list, 
  anchor.features = features
)

# integration anchors
anchors <- FindIntegrationAnchors(
  object.list = seu.list, 
  normalization.method = "SCT", 
  anchor.features = features
)

#integration
seu_integrated <- IntegrateData(anchorset = anchors, normalization.method = "SCT")

# PCA + UMAP
seu_integrated <- RunPCA(seu_integrated, verbose = FALSE)
seu_integrated <- RunUMAP(seu_integrated, dims = 1:30)
DimPlot(seu_integrated, group.by = "timepoint")
seu_integrated$timepoint <- factor(seu_integrated$timepoint, levels = c("0h", "6h", "48h", "2mo", "4mo"))
DimPlot(seu_integrated, group.by = "timepoint", split.by = "timepoint")

seu_integrated <- FindNeighbors(seu_integrated, dims = 1:10)
seu_integrated <- FindClusters(seu_integrated, resolution = 0.65)
DimPlot(seu_integrated)
DimPlot(seu_integrated, split.by = "timepoint")


seu_0h <- subset(seu_integrated, subset = timepoint == "0h")
seu_0h$cluster_integrated <- seu_0h$seurat_clusters
seu_0h <- RunPCA(seu_0h, verbose = FALSE)
seu_0h <- FindNeighbors(seu_0h, dims = 1:20)
seu_0h <- FindClusters(seu_0h, resolution = 0.6)
DimPlot(seu_0h)

table(
  integrated = seu_0h$cluster_integrated,
  new_0h = seu_0h$seurat_clusters
)
#Cluster 1, 5, 9 is enriched in Cluster 2 (Cluster S) of 0h

cluster_S_ids <- c("1", "5", "9")  # ?? seurat_clusters ??? character

seu_integrated$Cluster_S_label <- ifelse(
  seu_integrated$seurat_clusters %in% cluster_S_ids,
  "Cluster S",
  "Non-Cluster S"
)

seu_integrated$Cluster_S_label <- factor(
  seu_integrated$Cluster_S_label,
  levels = c("Cluster S", "Non-Cluster S")
)
DimPlot(seu_integrated, group.by="Cluster_S_label", split.by = "timepoint")

#save data
integrated_mat <- GetAssayData(
  seu_integrated,
  assay = "integrated",
  slot = "data"
)

write.csv(
  as.matrix(integrated_mat),
  file = "../Output/GSE249057_integrated_expression.csv"
)

meta_df <- seu_integrated@meta.data[, c("timepoint", "Cluster_S_label")]
write.csv(meta_df, file = "../Output/GSE249057_integrated_metadata.csv", row.names = TRUE)

save.image("../Output/GSE249057.RData")

