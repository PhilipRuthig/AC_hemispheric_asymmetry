' Statistics: bootstrapping from data, circular regression models on mice 
    individually, Sanity check: randomize data, Baysian plots, ... 

fit a circular regression model with the side as the explanatory variable 
for the dominant directions'

library(dplyr)
library(ggplot2)
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)
library(RcppArmadillo)
library(MASS)


# prepare the function to run the fit in chunks
bpnr_func <- function(sample, seed, its, b){
  fit <- bpnr(pred.I = domDir ~ side + layer,
              data = sample,
              its = its, burn = b, seed = seed)
  Intercept <- NISTradianTOdeg(fit$circ.coef.means)[1,]
  sider <- NISTradianTOdeg(fit$circ.coef.means)[2,]
  layerL4 <- NISTradianTOdeg(fit$circ.coef.means)[3,]
  layerL5 <- NISTradianTOdeg(fit$circ.coef.means)[4,]
  siderlayerL4 <- NISTradianTOdeg(fit$circ.coef.means)[5,]
  siderlayerL5 <- NISTradianTOdeg(fit$circ.coef.means)[6,]
  layerL4layerL5 <- NISTradianTOdeg(fit$circ.coef.means)[7,]
  beta1_1 <- fit$beta1[,1]
  beta1_2 <- fit$beta1[,2] 
  beta1_3 <- fit$beta1[,3]
  beta1_4 <- fit$beta1[,4] 
  beta2_1 <- fit$beta2[,1]
  beta2_2 <- fit$beta2[,2]
  beta2_3 <- fit$beta2[,3]
  beta2_4 <- fit$beta2[,4]
  model_fit <- fit(fit)[,1]
  return(list(Intercept, sider, layerL4, layerL5, siderlayerL4, siderlayerL5, layerL4layerL5, 
              beta1_1, beta1_2, beta1_3, beta1_4, beta2_1, beta2_2, beta2_3, beta2_4, model_fit))
}


loop_over_sampleID <- function(d, n, path, seed, its, b){
  sample.Intercept <- matrix(0, nrow = Nsim,ncol=5)
  sample.sider <- matrix(0, nrow = Nsim,ncol=5)
  sample.layerL4 <- matrix(0, nrow = Nsim,ncol=5)
  sample.layerL5 <- matrix(0, nrow = Nsim,ncol=5)
  sample.siderlayerL4 <- matrix(0, nrow = Nsim,ncol=5)
  sample.siderlayerL5 <- matrix(0, nrow = Nsim,ncol=5)
  sample.layerL4layerL5 <- matrix(0, nrow = Nsim,ncol=5)
  
  sample.beta1_1 <- matrix(0, nrow = Nsim,ncol=its)
  sample.beta1_2 <- matrix(0, nrow = Nsim,ncol=its)
  sample.beta1_3 <- matrix(0, nrow = Nsim,ncol=its)
  sample.beta1_4 <- matrix(0, nrow = Nsim,ncol=its)
  sample.beta2_1 <- matrix(0, nrow = Nsim,ncol=its)
  sample.beta2_2 <- matrix(0, nrow = Nsim,ncol=its)
  sample.beta2_3 <- matrix(0, nrow = Nsim,ncol=its)
  sample.beta2_4 <- matrix(0, nrow = Nsim,ncol=its)
  sample.fit <- matrix(0, nrow = Nsim,ncol=5)
  
  for (i in 1:Nsim){
    sample <- d %>%
      group_by(sampleID, side, layer) %>%
      slice_sample(n = n)
    out <- bpnr_func(sample, seed, its, b)
    sample.Intercept[i,] <- out[1][[1]]
    sample.sider[i,] <- out[2][[1]]
    sample.layerL4[i,] <- out[3][[1]]
    sample.layerL5[i,] <- out[4][[1]]
    sample.siderlayerL4[i,] <- out[5][[1]]
    sample.siderlayerL5[i,] <- out[6][[1]]
    sample.layerL4layerL5[i,] <- out[7][[1]]
 
    sample.beta1_1[i,] <- out[8][[1]]
    sample.beta1_2[i,] <- out[9][[1]]
    sample.beta1_3[i,] <- out[10][[1]]
    sample.beta1_4[i,] <- out[11][[1]]
    sample.beta2_1[i,] <- out[12][[1]]
    sample.beta2_2[i,] <- out[13][[1]]
    sample.beta2_3[i,] <- out[14][[1]]
    sample.beta2_4[i,] <- out[15][[1]]
    sample.fit[i,] <- out[16][[1]]
  }
  
  write.matrix(sample.Intercept, file=paste0(path, "bpnr2p_Intercept.csv"))
  write.matrix(sample.sider, file=paste0(path, "bpnr2p_sideR.csv"))
  write.matrix(sample.layerL4, file=paste0(path, "bpnr2p_layerL4.csv"))
  write.matrix(sample.layerL5, file=paste0(path, "bpnr2p_layerL5.csv"))
  write.matrix(sample.siderlayerL4, file=paste0(path, "bpnr2p_siderlayerL4.csv"))
  write.matrix(sample.siderlayerL5, file=paste0(path, "bpnr2p_siderlayerL5.csv"))
  write.matrix(sample.layerL4layerL5, file=paste0(path, "bpnr2p_layerL4layerL5.csv"))
  
  write.matrix(sample.fit, file=paste0(path, "bpnr2p_fit.csv"))
  write.matrix(sample.beta1_1, file=paste0(path, "bpnr2p_beta1_1.csv"))
  write.matrix(sample.beta1_2, file=paste0(path, "bpnr2p_beta1_2.csv"))
  write.matrix(sample.beta1_3, file=paste0(path, "bpnr2p_beta1_3.csv"))
  write.matrix(sample.beta1_4, file=paste0(path, "bpnr2p_beta1_4.csv"))
  write.matrix(sample.beta2_1, file=paste0(path, "bpnr2p_beta2_1.csv"))
  write.matrix(sample.beta2_2, file=paste0(path, "bpnr2p_beta2_2.csv"))
  write.matrix(sample.beta2_3, file=paste0(path, "bpnr2p_beta2_3.csv"))
  write.matrix(sample.beta2_4, file=paste0(path, "bpnr2p_beta2_4.csv"))
  
}




#MAIN
# prepare data
dominant_directions_fused <- read.table("...", quote="\"", comment.char="")
colnames(dominant_directions_fused)<-c("sampleID", "side", "layer", "z","y","x", "domDir")
data <- dominant_directions_fused[!(dominant_directions_fused$layer==0 |
                                      dominant_directions_fused$layer==4|
                                      dominant_directions_fused$layer==5),]
data$layer <- as.factor(data$layer)
attach(data)
rm(dominant_directions_fused)
data$domDir <- NISTdegTOradian(data$domDir)


ids = unique(data$sampleID)
set.seed(2024) # for reproducibility
Nsim = 25
its = 10000
b = 1000
seed = 2024

for (id in ids) {
  path = paste0("...", paste(id))
  dir.create(path)
  
  d <- data[(data$sampleID==id),]
  
  max_rows <- d %>%
    group_by(sampleID, side, layer) %>%
    summarise(max_rows = n())
  min_max_rows <- min(max_rows$max_rows)
  n <- as.integer(min_max_rows*0.05)
  
  loop_over_sampleID(d, n, path, seed, its, b)
}
