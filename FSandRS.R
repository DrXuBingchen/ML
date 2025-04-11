


feature_names <- colnames(train)

p_values <- numeric(length(feature_names))

for (i in seq_along(feature_names)) {
  feature <- train[[feature_names[i]]]
  anova_result <- aov(feature ~ as.factor(train_status$status)) 
  p_values[i] <- summary(anova_result)[[1]][["Pr(>F)"]][1]
}
fdr_threshold <- 0.05
adjusted_p <- p.adjust(p_values, method = "BH")
significant_features <- feature_names[adjusted_p < fdr_threshold]

train_filtered <- train[, significant_features, drop = FALSE]

final_data <- cbind(train_lable, train_filtered)

write.csv(final_data, "ANOVA_filtered_features.csv", row.names = FALSE)


train_filtered <- train[, significant_features]

normality_p <- sapply(train_filtered, function(x) {
  if (length(x) > 5000) {
    ks.test(x, "pnorm", mean = mean(x), sd = sd(x))$p.value
  } else {
    shapiro.test(x)$p.value
  }
})

is_normal <- normality_p > 0.05

n_features <- ncol(train_filtered)
cor_matrix <- matrix(NA, nrow = n_features, ncol = n_features)
colnames(cor_matrix) <- colnames(train_filtered)
rownames(cor_matrix) <- colnames(train_filtered)

for (i in 1:(n_features-1)) {
  for (j in (i+1):n_features) {
    x <- train_filtered[, i]
    y <- train_filtered[, j]

    if (is_normal[i] && is_normal[j]) {
      cor_method <- "pearson"
    } else {
      cor_method <- "spearman"
    }

    cor_matrix[i, j] <- cor(x, y, method = cor_method)
    cor_matrix[j, i] <- cor_matrix[i, j]  # 对称矩阵
  }
}

cor_df <- melt(cor_matrix, na.rm = TRUE) %>%
  filter(Var1 != Var2) %>%  
  filter(abs(value) > 0.9)  

cor_pairs <- unique(t(apply(cor_df[, 1:2], 1, sort)))

features_to_remove <- character(0)
remaining_features <- colnames(train_filtered)

for (i in 1:nrow(cor_pairs)) {
  pair <- cor_pairs[i, ]
  if (all(pair %in% remaining_features)) {

    counts <- table(cor_pairs)
    if (sum(pair[1] == cor_pairs) > sum(pair[2] == cor_pairs)) {
      remove_feature <- pair[1]
    } else {
      remove_feature <- pair[2]
    }
    features_to_remove <- union(features_to_remove, remove_feature)
    remaining_features <- setdiff(remaining_features, remove_feature)
  }
}

final_features <- train_filtered[, remaining_features]

final_data <- cbind(train_lable, final_features)

write.csv(final_data, "filtered_features_after_correlation.csv", row.names = FALSE)

x <- as.matrix(final_features)  
y <- train_status$status         

if (is.factor(y) && length(levels(y)) == 2) {
  family_type <- "binomial"    
} else if (is.numeric(y)) {
  family_type <- "gaussian"     
} else {
  stop("响应变量格式错误：需为二分类因子或连续型数值")
}

set.seed(123) 
lasso_cv <- cv.glmnet(
  x = x,
  y = y,
  family = family_type,
  alpha = 1,                    
  standardize = FALSE,          
  nfolds = 10,                  
  parallel = TRUE               
)


best_lambda <- lasso_cv$lambda.min
lasso_coef <- coef(lasso_cv, s = "lambda.min")

intercept_value <- as.numeric(lasso_coef["(Intercept)", ])
cat("最优lambda对应的截距项:", intercept_value, "\n\n")  

combined_data <- read.csv('combined_dataset.csv')
colnames(combined_data)
combined_data_RS <- combined_data[,11:24]
combined_data_label <- combined_data[,1:10]
xn = nrow(combined_data_RS)
yn = ncol(combined_data_RS)
combined_data_RS_matrix <- as.matrix(combined_data_RS)


coefPara <- coef(object = lasso_cv, s = 'lambda.min')



beta           = as.matrix(coefPara[which(coefPara != 0),])   
betai_Matrix   = as.matrix(beta[-1])                          
beta0_Matrix   = matrix(beta[1], xn, 1)                       
Radcore_Matrix = combined_data_RS_matrix %*% betai_Matrix + beta0_Matrix     
radscore_all   = as.numeric(Radcore_Matrix)
radscore_all1 = as.data.frame(radscore_all)

RadiomicsScore_all <- cbind(combined_data_RS, radscore_all1)
RadiomicsScore_all <- cbind(RadiomicsScore_all, combined_data_label)

write.csv(RadiomicsScore_all, "RadiomicsScore.csv", row.names=FALSE)













# 转换为数据框并过滤非零系数
coef_df <- data.frame(
  feature = rownames(lasso_coef),
  coefficient = as.numeric(lasso_coef)
) %>%
  filter(coefficient != 0) %>%
  filter(feature != "(Intercept)") %>%  # 移除截距项
  arrange(desc(abs(coefficient)))       # 按系数绝对值降序排序

# ------------------------------
# 步骤4：保存非零系数排序结果
# ------------------------------
write.csv(coef_df, "LASSO_Selected_Coefficients.csv", row.names = FALSE)

# ------------------------------
# 步骤5：绘制并保存LASSO结果图
# ------------------------------
# 图1：系数路径图
pdf("LASSO_Coefficient_Path.pdf", width = 8, height = 6)
plot(lasso_cv$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(best_lambda), lty = 2, col = "red")
title("")
dev.off()

# 图2：交叉验证误差曲线
pdf("LASSO_CV_Error.pdf", width = 8, height = 6)
plot(lasso_cv)
abline(v = log(best_lambda), lty = 2, col = "red")
title("")
dev.off()

# ------------------------------
# 步骤6：输出最终特征子集
# ------------------------------
selected_features <- coef_df$feature
final_lasso_data <- cbind(train_lable, final_features[, selected_features])
write.csv(final_lasso_data, "LASSO_Final_Features.csv", row.names = FALSE)

# ------------------------------
# 输出筛选信息
# ------------------------------
cat("LASSO筛选后保留特征数量:", length(selected_features), "\n")
cat("最大绝对系数:", max(abs(coef_df$coefficient)), "\n")
cat("结果已保存至：LASSO_Selected_Coefficients.csv 和 LASSO_Final_Features.csv\n")









