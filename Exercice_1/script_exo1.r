########### script_exo1.R ###########
# ce script execute des algorithmes de clustering sur le jeu de données Lsun
rm(list=ls())


########### Package ###########  
library(tidyverse)
library(igraph)
library(ggplot2)
library(kernlab)
library(mclust)
library(caret)


########################################## Importation du fichier ##########################################

P = read.csv(file = "../data/Lsun.lrn", header = FALSE, sep = "\t", skip = 4)[,-1]
label = read.csv(file = "../data/Lsun.cls", header = FALSE, sep = "\t",skip = 1)[,-c(1,3)]
data_P = data.frame(x = P[,1], y = P[,2],label = as.factor(label))

set.seed(1234)
P <- as.matrix(P)

########################################## graphe objectif  ########################################## 
g <- ggplot(data = data_P, aes(x=x, y =y)) + geom_point(aes(col = label))
g

########################################## clusteirng specteral  ########################################## 

spec_poly <- specc(P, centers=3, kernel="polydot", kpar=list(degree=2))
confusionMatrix(factor(label), factor(spec_poly@.Data))


spec_rad <- specc(P, centers=2, kernel="rbfdot")
confusionMatrix(factor(label), factor(spec_rad@.Data))


########################################## clustering kmeans a noyau  ########################################## 

kkmeans_rad <- kkmeans(P, centers=3, kernel="rbfdot", nstart=20)
confusionMatrix(factor(label), factor(kkmeans_rad@.Data))


kkmeans_poly <- kkmeans(P, centers=3, kernel="polydot", kpar=list(degree=2), nstart=20)
confusionMatrix(factor(label), factor(kkmeans_poly@.Data))


########################################## clustering kmeans classique  ########################################## 

kmeans_classic <- kmeans(P, centers=3)


########################################## graphe resultat ########################################## 

df2 <- data_P %>% mutate(spec.poly = as.factor(spec_poly@.Data),
                     spec.rad = as.factor(spec_rad@.Data), kkmeans.rad = as.factor(kkmeans_rad@.Data),
                     kkmeans.poly = as.factor(kkmeans_poly@.Data),
                     Kmeans.classic = as.factor(kmeans_classic$cluster)) %>%
  pivot_longer(-c(x,y), names_to = "Methode", values_to="Label")

ggplot(df2) + aes(x=x, y=y, color=Label) + geom_point() + facet_wrap(~Methode)


########################################## resultat tableau ########################################## 


ari_kmeans <- function(k){
  res_clus <- kmeans(P, centers = k, nstart = 20)
  return(adjustedRandIndex(res_clus$cluster, label))
}

ari_spec <- function(k){
  res_clus <- specc(P, centers=k, kernel="rbfdot")
  return(adjustedRandIndex(res_clus@.Data, label))
}

ari_kmeans_kernel <- function(k){
  res_clus <- kkmeans(P, centers=k, kernel="rbfdot", nstart=20)
  return(adjustedRandIndex(res_clus@.Data, label))
}


sapply(2:5, ari_kmeans)
sapply(2:5, ari_kmeans_kernel)
sapply(2:5, ari_spec)



# Note avg silhouette
# ss <- silhouette(kmeans_classic$cluster, dist(P))
# mean(ss[, 3])
# 
# ss <- silhouette(spec_rad@.Data, dist(P))
# mean(ss[, 3])



########################################## graphe reslustats clustering ########################################## 

# specteral

normalize <- function(x){
  return(x/sqrt(sum(x^2))) 
}


mat_simi <- kernelMatrix(rbfdot(), P)

G <- graph_from_adjacency_matrix(as.matrix(dist(mat_simi)))

L <- laplacian_matrix(G, sparse=F, norm=T)
spec <- eigen(L)

df <- tibble(vp=1:6,valeur=rev(spec$values)[1:6])

ggplot(df)+aes(x=vp,y=valeur)+geom_point()

U <- spec$vectors[,1:5]
U.norm <- t(apply(U,1,normalize)) 

res <- kmeans(U.norm, 3, nstart=20)


# kkmeans variance intra mais varie
plot(kkmeans(P, centers=6, kernel="polydot", kpar=list(degree=2), nstart=20)@withinss,type="b",xlab="nb groupes",ylab="inertie intra")

