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

# prepare data
dominant_directions_fused <- read.table("", quote="\"", comment.char="")
colnames(dominant_directions_fused)<-c("sampleID", "side", "layer", "z","y","x", "domDir")
data <- dominant_directions_fused[!(dominant_directions_fused$layer==0 |
                                      dominant_directions_fused$layer==4|
                                      dominant_directions_fused$layer==5),]
data$layer <- as.factor(data$layer)
attach(data)
rm(dominant_directions_fused)
data$domDir <- NISTdegTOradian(data$domDir)

# prepare the function to run the fit in chunks
bpnr_func <- function(sample, seed, its, b){
  fit <- bpnr(pred.I = domDir ~ side,
              data = sample,
              its = its, burn = b, seed = seed)
  Intercept <- NISTradianTOdeg(fit$circ.coef.means)[1,]
  sider <- NISTradianTOdeg(fit$circ.coef.means)[2,]
  beta1_1 <- fit$beta1[,1] # Intercept
  beta1_2 <- fit$beta1[,2] 
  beta2_1 <- fit$beta2[,1]
  beta2_2 <- fit$beta2[,2]
  model_fit <- fit(fit)[,1]
  return(list(Intercept, sider, beta1_1, beta1_2, beta2_1, beta2_2, model_fit))
}

set.seed(2024)
Nsim = 25
its = 10000
b = 1000
seed = 2024

sample.Intercept <- matrix(0, nrow = Nsim,ncol=5)
sample.sider <- matrix(0, nrow = Nsim,ncol=5)
sample.beta1_1 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta1_2 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta2_1 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta2_2 <- matrix(0, nrow = Nsim,ncol=its)
sample.fit <- matrix(0, nrow = Nsim,ncol=5)

max_rows <- data %>%
  group_by(sampleID, side, layer) %>%
  summarise(max_rows = n())
min_max_rows <- min(max_rows$max_rows)
n <- as.integer(min_max_rows*0.1)

for (i in 1:Nsim){
  sample <- data %>%
    group_by(sampleID, side, layer) %>%
    slice_sample(n = n)
  out <- bpnr_func(sample, seed, its, b)
  sample.Intercept[i,] <- out[1][[1]]
  sample.sider[i,] <- out[2][[1]]
  sample.beta1_1[i,] <- out[3][[1]]
  sample.beta1_2[i,] <- out[4][[1]]
  sample.beta2_1[i,] <- out[5][[1]]
  sample.beta2_2[i,] <- out[6][[1]]
  sample.fit[i,] <- out[7][[1]]
}

write.matrix(sample.Intercept, file=".../bpnr1p_Intercept.csv")
write.matrix(sample.sider, file=".../bpnr1p_sideR.csv")
write.matrix(sample.fit, file=".../bpnr1p_fit.csv")
write.matrix(sample.beta1_1, file=".../bpnr1p_beta1_1.csv")
write.matrix(sample.beta1_2, file=".../bpnr1p_beta1_2.csv")
write.matrix(sample.beta2_1, file=".../bpnr1p_beta2_1.csv")
write.matrix(sample.beta2_2, file=".../bpnr1p_beta2_2.csv")
