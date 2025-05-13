
# 加载必要的包
library(mclust)
library(tidyverse)
library(factoextra)
library(readxl)
library(openxlsx)
library(scatterplot3d)
library(ggplot2)

# 读取数据
data <- read_xlsx("../../3.基线数据分析/2.聚类分析/2.GMM/校正并标准化后的患者行为变量数据.xlsx", sheet = 1)

# 选择变量
EFs_data <- data %>%
  select(Stroop_incongruent_rt, 
         `Stroop_interference effect_rt`,
         Nogo_acc, 
         Switch_cost, 
         `RM-1,750_acc`,
         `RM-750_acc`, 
         DSBT_Span)

model_names <- c("EII", "VII", "EEI", "VEI", "EVI", "VVI", "EEE", "VEE", "EVE", "VVE", "EEV", "VEV", "EVV", "VVV")
# 初始化存储 BIC 值和模型的变量
bic_values <- data.frame(Cluster_Number = integer(), Model = character(), BIC = numeric())
models <- list()

set.seed(24)  # 设置随机种子，确保结果可重复
options(warn = -1)  # 忽略警告信息

# 在原始数据上执行 GMM 聚类，聚类数量范围为 2 到 7
for (k in 2:7) {
  for (model_name in model_names) {
    model <- Mclust(EFs_data, G = k, modelNames = model_name, control = emControl(itmax = 1000))
    models[[paste(model_name, k, sep = "_")]] <- model  # 存储模型
    bic_values <- rbind(bic_values, data.frame(
      Cluster_Number = k,
      Model = model_name,
      BIC = max(model$BIC, na.rm = TRUE)
    ))
  }
}

# 基于 BIC 选择最佳模型
best_model_row <- which.max(bic_values$BIC)
best_k <- bic_values$Cluster_Number[best_model_row]
best_model_name <- bic_values$Model[best_model_row]
best_model <- models[[paste(best_model_name, best_k, sep = "_")]]
cat("Best model type:", best_model_name, "\nBest number of clusters:", best_k, "\n")

# 绘制 BIC 值对比图
png("BIC_Comparison_Plot.png", width = 1600, height = 1200, res = 300)
bic_plot <- ggplot(bic_values, aes(x = Cluster_Number, y = BIC, color = Model, group = Model)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(title = "BIC Comparison Across Models and Cluster Numbers",
       x = "Number of Clusters",
       y = "BIC",
       color = "Model") +
  theme_minimal()
print(bic_plot)
dev.off()

# 保存 BIC 数据到 Excel
write.xlsx(bic_values, "BIC_Comparison_Results.xlsx", overwrite = TRUE)

# 获取聚类标签并调整为从 0 开始
optimal_cluster_labels <- best_model$classification - 1
EFs_with_optimal_clusters <- EFs_data %>%
  mutate(Cluster = factor(optimal_cluster_labels))

# 对聚类结果进行 PCA 降维到 3D
pca_result <- prcomp(EFs_data, center = TRUE, scale. = TRUE)
pca_data <- as.data.frame(pca_result$x[, 1:3])  # 取前 3 个主成分
pca_data$Cluster <- EFs_with_optimal_clusters$Cluster  # 添加聚类标签

# 使用 scatterplot3d 生成静态 3D 散点图，并指定高分辨率像素
png("3D_PCA_Clustering_Visualization_NatureStyle.png", width = 4000, height = 4000, res = 300)
s3d <- scatterplot3d(
  pca_data$PC1,
  pca_data$PC2,
  pca_data$PC3,
  color = as.numeric(pca_data$Cluster) + 1,
  pch = 19,
  angle = 45,
  xlab = "Principal Component 1",
  ylab = "Principal Component 2",
  zlab = "Principal Component 3",
  main = "3D PCA Visualization of Clustering Results",
  grid = FALSE,
  box = TRUE
)
dev.off()

# 打印完成消息
cat("BIC 比较图和 PCA 降维后的静态 3D 可视化完成\n")


# 确定各模型的最佳聚类数
best_EII <- bic_values %>% filter(Model == "EII") %>% slice(which.max(BIC))
best_VII <- bic_values %>% filter(Model == "VII") %>% slice(which.max(BIC))
best_EEI <- bic_values %>% filter(Model == "EEI") %>% slice(which.max(BIC))
best_VEI <- bic_values %>% filter(Model == "VEI") %>% slice(which.max(BIC))
best_EVI <- bic_values %>% filter(Model == "EVI") %>% slice(which.max(BIC))
best_VVI <- bic_values %>% filter(Model == "VVI") %>% slice(which.max(BIC))
best_EEE <- bic_values %>% filter(Model == "EEE") %>% slice(which.max(BIC))
best_VEE <- bic_values %>% filter(Model == "VEE") %>% slice(which.max(BIC))
best_EVE <- bic_values %>% filter(Model == "EVE") %>% slice(which.max(BIC))
best_VVE <- bic_values %>% filter(Model == "VVE") %>% slice(which.max(BIC))
best_EEV <- bic_values %>% filter(Model == "EEV") %>% slice(which.max(BIC))
best_VEV <- bic_values %>% filter(Model == "VEV") %>% slice(which.max(BIC))
best_EVV <- bic_values %>% filter(Model == "EVV") %>% slice(which.max(BIC))
best_VVV <- bic_values %>% filter(Model == "VVV") %>% slice(which.max(BIC))

# 提取最佳模型的聚类标签
EII_model <- models[[paste("EII", best_EII$Cluster_Number, sep = "_")]]
VII_model <- models[[paste("VII", best_VII$Cluster_Number, sep = "_")]]
EEI_model <- models[[paste("EEI", best_EEI$Cluster_Number, sep = "_")]]
VEI_model <- models[[paste("VEI", best_VEI$Cluster_Number, sep = "_")]]
EVI_model <- models[[paste("EVI", best_EVI$Cluster_Number, sep = "_")]]
VVI_model <- models[[paste("VVI", best_VVI$Cluster_Number, sep = "_")]]
EEE_model <- models[[paste("EEE", best_EEE$Cluster_Number, sep = "_")]]
VEE_model <- models[[paste("VEE", best_VEE$Cluster_Number, sep = "_")]]
EVE_model <- models[[paste("EVE", best_EVE$Cluster_Number, sep = "_")]]
VVE_model <- models[[paste("VVE", best_VVE$Cluster_Number, sep = "_")]]
EEV_model <- models[[paste("EEV", best_EEV$Cluster_Number, sep = "_")]]
VEV_model <- models[[paste("VEV", best_VEV$Cluster_Number, sep = "_")]]
EVV_model <- models[[paste("EVV", best_EVV$Cluster_Number, sep = "_")]]
VVV_model <- models[[paste("VVV", best_VVV$Cluster_Number, sep = "_")]]

# 调整聚类标签为从 0 开始
EII_labels <- EII_model$classification - 1
VII_labels <- VII_model$classification - 1
EEI_labels <- EEI_model$classification - 1
VEI_labels <- VEI_model$classification - 1
EVI_labels <- EVI_model$classification - 1
VVI_labels <- VVI_model$classification - 1
EEE_labels <- EEE_model$classification - 1
VEE_labels <- VEE_model$classification - 1
EVE_labels <- EVE_model$classification - 1
VVE_labels <- VVE_model$classification - 1
EEV_labels <- EEV_model$classification - 1
VEV_labels <- VEV_model$classification - 1
EVV_labels <- EVV_model$classification - 1
VVV_labels <- VVV_model$classification - 1

# 保存结果到 Excel 文件
output_file <- "Best_Cluster_Labels_All_Models.xlsx"  # 更新输出文件名
wb <- createWorkbook()

# 添加 EII 的聚类标签
addWorksheet(wb, "EII_Best_Cluster")
writeData(wb, "EII_Best_Cluster", data.frame(ID = 1:length(EII_labels), Cluster_Label = EII_labels))

# 添加 VII 的聚类标签
addWorksheet(wb, "VII_Best_Cluster")
writeData(wb, "VII_Best_Cluster", data.frame(ID = 1:length(VII_labels), Cluster_Label = VII_labels))

# 添加 EEI 的聚类标签
addWorksheet(wb, "EEI_Best_Cluster")
writeData(wb, "EEI_Best_Cluster", data.frame(ID = 1:length(EEI_labels), Cluster_Label = EEI_labels))

# 添加 VEI 的聚类标签
addWorksheet(wb, "VEI_Best_Cluster")
writeData(wb, "VEI_Best_Cluster", data.frame(ID = 1:length(VEI_labels), Cluster_Label = VEI_labels))

# 添加 EVI 的聚类标签
addWorksheet(wb, "EVI_Best_Cluster")
writeData(wb, "EVI_Best_Cluster", data.frame(ID = 1:length(EVI_labels), Cluster_Label = EVI_labels))

# 添加 VVI 的聚类标签
addWorksheet(wb, "VVI_Best_Cluster")
writeData(wb, "VVI_Best_Cluster", data.frame(ID = 1:length(VVI_labels), Cluster_Label = VVI_labels))

# 添加 EEE 的聚类标签
addWorksheet(wb, "EEE_Best_Cluster")
writeData(wb, "EEE_Best_Cluster", data.frame(ID = 1:length(EEE_labels), Cluster_Label = EEE_labels))

# 添加 VEE 的聚类标签
addWorksheet(wb, "VEE_Best_Cluster")
writeData(wb, "VEE_Best_Cluster", data.frame(ID = 1:length(VEE_labels), Cluster_Label = VEE_labels))

# 添加 EVE 的聚类标签
addWorksheet(wb, "EVE_Best_Cluster")
writeData(wb, "EVE_Best_Cluster", data.frame(ID = 1:length(EVE_labels), Cluster_Label = EVE_labels))

# 添加 VVE 的聚类标签
addWorksheet(wb, "VVE_Best_Cluster")
writeData(wb, "VVE_Best_Cluster", data.frame(ID = 1:length(VVE_labels), Cluster_Label = VVE_labels))

# 添加 EEV 的聚类标签
addWorksheet(wb, "EEV_Best_Cluster")
writeData(wb, "EEV_Best_Cluster", data.frame(ID = 1:length(EEV_labels), Cluster_Label = EEV_labels))

# 添加 VEV 的聚类标签
addWorksheet(wb, "VEV_Best_Cluster")
writeData(wb, "VEV_Best_Cluster", data.frame(ID = 1:length(VEV_labels), Cluster_Label = VEV_labels))

# 添加 EVV 的聚类标签
addWorksheet(wb, "EVV_Best_Cluster")
writeData(wb, "EVV_Best_Cluster", data.frame(ID = 1:length(EVV_labels), Cluster_Label = EVV_labels))

# 添加 VVV 的聚类标签
addWorksheet(wb, "VVV_Best_Cluster")
writeData(wb, "VVV_Best_Cluster", data.frame(ID = 1:length(VVV_labels), Cluster_Label = VVV_labels))

# 保存 Excel 文件
saveWorkbook(wb, output_file, overwrite = TRUE)

# 打印完成消息
cat("所有模型的最佳聚类标签已存储至", output_file, "\n")




# 加载必要的包
library(mclust)
library(tidyverse)
library(readxl)
library(openxlsx)
library(aricode)

# 定义执行自助抽样和聚类的函数
perform_bootstrap_clustering <- function(data, model_names, n_clusters_range = 2:7, 
                                         n_bootstrap = 300, remove_percent = 0.15) {
  n_samples <- nrow(data)
  n_remove <- floor(n_samples * remove_percent)
  
  # 初始化结果存储数据框
  results <- data.frame(
    Model = character(),
    Cluster_Number = integer(),
    ARI_Mean = numeric(),
    ARI_SD = numeric(),
    ARI_CI_Lower = numeric(),
    ARI_CI_Upper = numeric()
  )
  
  # 对每个模型和每个聚类数进行自助抽样分析
  for (model_name in model_names) {
    for (n_clusters in n_clusters_range) {
      # 获取原始聚类结果
      original_model <- Mclust(data, G = n_clusters, modelNames = model_name)
      original_labels <- original_model$classification
      
      # 初始化当前配置的ARI值存储
      ari_values <- numeric(n_bootstrap)
      
      # 执行自助抽样
      for (i in 1:n_bootstrap) {
        # 随机选择要保留的样本
        keep_indices <- sample(1:n_samples, n_samples - n_remove)
        subsample_data <- data[keep_indices, ]
        
        # 对子样本进行聚类
        tryCatch({
          subsample_model <- Mclust(subsample_data, G = n_clusters, 
                                    modelNames = model_name, 
                                    control = emControl(itmax = 100))
          
          # 计算ARI
          if (!is.null(subsample_model)) {
            # 提取原始标签和新标签（仅对应于保留的样本）
            original_subset <- original_labels[keep_indices]
            new_labels <- subsample_model$classification
            
            # 计算ARI
            ari_values[i] <- adjustedRandIndex(original_subset, new_labels)
          }
        }, error = function(e) {
          ari_values[i] <- NA
        })
      }
      
      # 计算平均ARI、标准差和置信区间
      mean_ari <- mean(ari_values, na.rm = TRUE)
      sd_ari <- sd(na.omit(ari_values))  # 标准差
      ci_lower <- mean_ari - 1.96 * (sd_ari / sqrt(n_bootstrap))  # 95% 置信区间下限
      ci_upper <- mean_ari + 1.96 * (sd_ari / sqrt(n_bootstrap))  # 95% 置信区间上限
      
      # 存储结果
      results <- rbind(results, data.frame(
        Model = model_name,
        Cluster_Number = n_clusters,
        ARI_Mean = mean_ari,
        ARI_SD = sd_ari,
        ARI_CI_Lower = ci_lower,
        ARI_CI_Upper = ci_upper
      ))
    }
  }
  
  return(results)
}

# 读取数据
data <- read_xlsx("/Users/zhangtongyi/Desktop/我论文的数据文档/P0032024-认知聚类-project(最新版）/1.当前最新版本 /3.基线数据分析/2.聚类分析/2.GMM/校正并标准化后的患者行为变量数据.xlsx", sheet = 1)

# 选择变量
EFs_data <- data %>%
  select(Stroop_incongruent_rt,
         `Stroop_interference effect_rt`,
         Nogo_acc,
         Switch_cost,
         `RM-1,750_acc`,
         `RM-750_acc`,
         DSBT_Span)

# 执行自助抽样分析
set.seed(24)  # 设置随机种子以确保可重复性
model_names <- c("EVI")
stability_results <- perform_bootstrap_clustering(EFs_data, model_names)

# 将结果按 Model 和 Cluster_Number 排序
stability_results <- stability_results %>%
  arrange(Model, Cluster_Number)

# 保存结果到 Excel
write.xlsx(stability_results, "Clustering_Stability_Results_Extended.xlsx", 
           rowNames = FALSE, overwrite = TRUE)

# 打印结果
print(stability_results)

