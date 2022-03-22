########### B-spline.R ###########
# Ce script effectue un clustering fonctionel sur .
rm(list=ls())


########### Package ########### 
library(tidyverse)
library(mclust)
library(caret)
library(fda)


########################################## Importation du fichier ##########################################

# Les données issue de la fonction de densité
data <- read.csv("../data/StarLightCurves_TRAIN.tsv", sep="\t", header = F)
data <- as.matrix(data)
X <- data[,2:ncol(data)]
y <- data[,1]
int_T <- seq(0, ncol(data)-2)


# graphe objectif 
matplot(int_T, t(X), type = "l", lwd=2, xlab="", ylab="", col=y)


########################################## Approximation par B-splines  ##########################################

####### création de la base B-splines 
# Choix des noeuds 
noeuds <- seq(0, ncol(data)-1, length.out = 20)


# Base B-splines
base_bspline <- create.bspline.basis(c(0,max(int_T)), breaks = noeuds, norder=4)


# Approximation 
X_hat <- smooth.basis(int_T, t(X), base_bspline)

# graphes de nos courbes approximmer
plot(X_hat)


########################################## Analyse en composante fonctionnel   ##########################################
pca_courbe <- pca.fd(X_hat$fd, nharm = 4)

## Pourcentage d'inertie expliquer
pca_courbe$varprop

# score de l'ACPF avec plus 80% d'inertie expliquer par les 3 premieres fonctions propres
score_pca_courbe <- pca_courbe$scores[, 1:3]

# graphe ACPF sur les deux premier axes
plot(score_pca_courbe[,1], score_pca_courbe[,2], pch=15, col=y)


########################################## Clustering fonctionnel 2 step ##########################################

######################### Coefficient #########################
#### Coefficient + Kmeans 
coef_kmeans <- kmeans(t(X_hat$fd$coefs), centers=3, nstart=50)

# evoulution de la variance intra 
I.intra <- sapply(1:10, FUN=function(k) kmeans(t(X_hat$fd$coefs), centers=k, nstart=50)$tot.withinss)
plot(I.intra,type="b",xlab="nb groupes",ylab="inertie intra")

# taux de reussite
table(y, coef_kmeans$cluster)
adjustedRandIndex(coef_kmeans$cluster, y)

# resultat graphique
matplot(int_T, t(X), type = "l", lwd=2, xlab="", ylab="", col=coef_kmeans$cluster)


#### ACPF + Kmeans 
acpf_kmeans <- kmeans(score_pca_courbe, centers = 3, nstart = 50)

# evoulution de la variance intra 
I.intra <- sapply(1:10,FUN=function(k) kmeans(score_pca_courbe, centers=k, nstart=50)$tot.withinss)
plot(I.intra, type="l", xlab="nb groupes", ylab="inertie intra")

# graphe acpf avec lescluster trouve 
plot(score_pca_courbe[,1], score_pca_courbe[,2], pch=15, col= acpf_kmeans$cluster)

# resultat graphique
matplot(int_T, t(X), type = "l", lwd=2, xlab="", ylab="", col=acpf_kmeans$cluster)

# taux de reussite
table(y, acpf_kmeans$cluster)
adjustedRandIndex(acpf_kmeans$cluster, y)

