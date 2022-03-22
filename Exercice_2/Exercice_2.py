# package
import warnings
from gudhi.clustering.tomato import Tomato
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from numba_ops import sdtw_div
from scipy.spatial import distance
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import rand_score, silhouette_score


# Fonction mis en place
def dist_eucli(a, b):
    return distance.euclidean(a.tolist(), b.tolist())


def dist_sdtw(a, b):
    return sdtw_div(a.reshape(a.shape[0], 1), b.reshape(b.shape[0], 1), gamma=0.1)


def kmeans_sdtw(X, K, dist_funct, itermax=10, nstart=1):

    opt_cost = float('inf')
    n = X.shape[0]
    d = X.shape[1]

    Centres = None
    color = None
    opt_centre = None
    opt_color = None
    costt = None

    for nstart_id in range(nstart):

        old_Centres = np.zeros((K, d))
        Centres = X[np.random.choice(range(n), K), :]  # Tableau des centres.
        color = np.zeros((n, ))  # Donne le groupe du point i.
        distance_min = np.zeros((n,))   # Renvoie le carré de la distance au centre du groupe.
        centre_gardes = np.array([True for i in range(K)])

        Nstep = 1
        for j in range(n):
            cost = float('inf')
            best_ind = 1
            for i in range(K):
                newcost = dist_funct(X[j, :], Centres[i, :]) # distance de series temporelle
                if newcost < cost:
                    cost = newcost
                    best_ind = i

            color[j] = best_ind
            distance_min[j] = cost

        while int(np.sum(old_Centres != Centres)) > 0 and Nstep < itermax:
            Nstep = Nstep + 1
            old_Centres = Centres

            # Etape 1 : Mise a jour de Centres
            for i in range(K):
                nb_points_cloud = (color == i).sum()
                if nb_points_cloud > 1:
                    Centres[i, ] = X[color == i, ].mean(axis=0)
                    # Centres[i, ] = performDBA(X[color == i, ])
                elif nb_points_cloud == 1:
                    Centres[i, ] = X[color == i, ]
                else:
                    centre_gardes[i] = False

            # Etape 2 : Mise a jour de color et distance_min

            for j in range(n):
                cost = float('inf')
                best_ind = 1
                for i in range(K):
                    if centre_gardes[i]:
                        newcost = dist_funct(X[j, :], Centres[i, :])
                        if newcost < cost:
                            cost = newcost
                            best_ind = i

                color[j] = best_ind
                distance_min[j] = cost

            costt = distance_min.mean()

    if costt <= opt_cost:
        opt_cost = costt
        opt_centre = Centres
        opt_color = color

    return {'cluster': opt_color, 'centres': opt_centre, 'cost': opt_cost}


def aggrega(X, pas=3):
    new_X = []
    for i in range(0, X.shape[1], pas):
        new_X.append(X[:, i:(i+pas)].mean(axis=1).tolist())

    new_X = np.array(new_X).transpose()
    return new_X


def plot_clust(X, y):
    dico_color = {1: 'black', 2: 'blue', 3: 'red', 4: 'green', 5: 'pink', 6: 'yellow'}
    for ind in range(X.shape[0]):
        plt.plot(range(X.shape[1]), X[ind, :], color=dico_color[y[ind]], linewidth=1)


def graphe(X, dist_func):
    l_graphe = list()
    for couple in combinations(range(X.shape[0]), 2):
        val = dist_func(X[couple[0], :], X[couple[1], :])
        l_graphe.append((str(couple[0]), str(couple[1]), val))

    G = nx.Graph()
    G.add_weighted_edges_from(l_graphe)
    return G


def matrix_conf(a, b):
    a = pd.Series(a, name='Label')
    b = pd.Series(b, name='Prediction')
    return pd.crosstab(a, b)


# Programme principal
# Importation
X = np.loadtxt('../data/StarLightCurves_TRAIN.tsv', usecols=range(1, 1025))
y = np.loadtxt('../data/StarLightCurves_TRAIN.tsv', usecols=0)
print(X.shape)
print(y.shape)

# On prend un échantillon et on aggrége nos courbes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, stratify=y, random_state=13)
X_test = aggrega(X_test, pas=4)
print(X_test.shape)


# graphe objectif
plot_clust(X_test, y_test)
plt.xlabel("Time")
plt.ylabel("Valeur")
plt.savefig('../img/vrai_label.png')
plt.show()

plt.figure(figsize=(9, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    dico_color = {1: 'black', 2: 'blue', 3: 'red'}
    plt.plot(range(X_test.shape[1]), X_test[y_test == i+1].mean(axis=0), color=dico_color[i+1])
    plt.ylim(-1.5, 3.5)
    plt.title(f"Groupe {i+1}, nb={len(X_test[y_test == i+1])}")
plt.savefig('../img/courbes_labels.png')
plt.show()


# calcul matrice distance
G = graphe(X_test, dist_sdtw)
mat_dist = nx.to_numpy_array(G)


# kmeans avec dist sdtw
plt.figure(figsize=(10, 10))
for i, k in enumerate([2, 3, 4, 5]):
    plt.subplot(2, 2, i + 1)
    dico_res = kmeans_sdtw(X_test, k, dist_sdtw, 15, 10)
    plot_clust(X_test, dico_res['cluster']+1)
    plt.title("k=%d, r=%.2f, s=%.2f" %
              (k, rand_score(y_test, dico_res['cluster']+1),
               silhouette_score(mat_dist, dico_res['cluster']+1, metric='precomputed')))
    print(matrix_conf(y_test, dico_res['cluster']+1))
plt.savefig('../img/kmeans_sdtw.png')
plt.show()


# Specteral avec matrice distance dist sdtw
adj_mat = nx.adjacency_matrix(G)

plt.figure(figsize=(10, 10))
for i, k in enumerate([2, 3, 4, 5]):
    plt.subplot(2, 2, i + 1)
    model = SpectralClustering(k, affinity='precomputed', n_init=100)
    model.fit(adj_mat)
    y_pred = model.labels_+1
    plot_clust(X_test, y_pred)
    plt.title("k=%d, r=%.2f, s=%.2f" %
              (k, rand_score(y_test, y_pred),
               silhouette_score(mat_dist, y_pred, metric='precomputed')))
    print(matrix_conf(y_test, y_pred))

plt.savefig('../img/specteral_dist_sdtw.png')
plt.show()


# Clustering heirarchique
mat_dist = nx.to_numpy_array(G)

plt.figure(figsize=(10, 10))
for i, k in enumerate([2, 3, 4, 5]):
    plt.subplot(2, 2, i + 1)
    model = AgglomerativeClustering(k, affinity='precomputed', linkage='average')
    model.fit(mat_dist)
    y_pred = model.labels_+1
    plot_clust(X_test, y_pred)
    plt.title("k=%d, r=%.2f, s=%.2f" %
              (k, rand_score(y_test, y_pred),
               silhouette_score(mat_dist, y_pred, metric='precomputed')))
    print(matrix_conf(y_test, y_pred))

plt.savefig('../img/hclust_dist_sdtw.png')
plt.show()


# kmeans avec distance euclidienne
plt.figure(figsize=(10, 10))
for i, k in enumerate([2, 3, 4, 5]):
    plt.subplot(2, 2, i + 1)
    model = KMeans(n_clusters=k)
    model.fit(X_test)
    y_pred = model.predict(X_test)+1
    plot_clust(X_test, y_pred)
    plt.title("k=%d, r=%.2f, s=%.2f" %
              (k, rand_score(y_test, y_pred),
               silhouette_score(X_test, y_pred)))

    print(matrix_conf(y, y_pred))

plt.savefig('../img/kmeans_eucli.png')
plt.show()


# Tomato
model = Tomato(k=5, density_type='logDTM')
model = model.fit(X_test)
warnings.filterwarnings('ignore')
model.plot_diagram()

plt.figure(figsize=(10, 10))
for i, k in enumerate([2, 3, 4, 5]):
    plt.subplot(2, 2, i + 1)
    model.n_clusters_ = k
    y_pred = model.labels_+1
    plot_clust(X_test, y_pred)
    plt.title("k=%d, r=%.2f, s=%.2f" %
              (k, rand_score(y_test, y_pred),
               silhouette_score(mat_dist, y_pred, metric='precomputed')))
    print(matrix_conf(y_test, y_pred))

plt.savefig('../img/Tomato_clust.png')
plt.show()


# Tomato all
model = Tomato(density_type="logKDE")
model = model.fit(X)
warnings.filterwarnings('ignore')
model.plot_diagram()
print(model.n_clusters_)


plt.figure(figsize=(10, 10))
for i, k in enumerate([2, 3, 4, 5]):
    plt.subplot(2, 2, i + 1)
    model.n_clusters_ = k
    y_pred = model.labels_+1
    plot_clust(X, y_pred)
    plt.title("k=%d, r=%.2f" %
              (k, rand_score(y, y_pred)))
    print(matrix_conf(y, y_pred))

plt.savefig('../img/Tomato_clust_all.png')
plt.show()
