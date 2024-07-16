import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, random, operator
import time
from sklearn.metrics import pairwise_distances

# Maximum number of iterations
MAX_ITER = 100

# khoi tao ma tran thanh vien
def initializeMembershipMatrix(n, k):  # initializing the membership matrix
    membership_mat = []
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x / summation for x in random_num_list]

        flag = temp_list.index(max(temp_list))
        for j in range(0, len(temp_list)):
            if (j == flag):
                temp_list[j] = 1
            else:
                temp_list[j] = 0

        membership_mat.append(temp_list)
    return membership_mat

# Tinh toan tam cum
def calculateClusterCenter(membership_mat, dataset, n, k, m):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [p ** m for p in x]
        denominator = sum(xraised)
        temp_num = []
        for i in range(n):
            data_point = list(dataset.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num)))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers

# cap nhat gia tri thanh vien
def updateMembershipValue(cluster_centers, n, dataset, k, m): # Updating the membership value
    p = float(2/(m-1))
    arr = [[0.0 for _ in range(k)] for _ in range(n)]
    for i in range(n):
        x = list(dataset.iloc[i])
        distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            arr[i][j] = float(1/den)
    return arr

# lay cac tam cum
def getClusters(membership_mat, n): # getting the clusters
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

# dieu kien dung
def calculate_norm(U_new, U_old):
    norm = 0.0
    for i in range(len(U_new)):
        for j in range(len(U_new[i])):
            norm += (U_new[i][j] - U_old[i][j]) ** 2
    return np.sqrt(norm)

# thuat toan FCM voi tam cum ban dau la cac diem du lieu ngau nhien
def fuzzyCMeansClustering(n, dataset, k, epsilon, m):  # Third iteration Random vectors from data

    # Membership Matrix
    membership_mat = initializeMembershipMatrix(n, k)
    curr = 0
    acc = []
    while curr < MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat, dataset, n, k, m)
        membership_mat_new = updateMembershipValue(cluster_centers,  n, dataset, k, m)

        if calculate_norm(membership_mat_new, membership_mat) < epsilon:
            break

        membership_mat = membership_mat_new
        cluster_labels = getClusters(membership_mat, n)
        acc.append(cluster_labels)
        curr += 1

    return cluster_labels, cluster_centers, acc

# def star():
def star(k, m, epsilon, file_path):

    # file_path = "C:\\GR1\\data.csv"
    dataset1 = pd.read_csv(file_path)

    # Number of data points
    n1 = len(dataset1)

    start_time = time.time()
    labels, centers, acc = fuzzyCMeansClustering(n1, dataset1, k, epsilon, m)
    end_time = time.time()

    execution_time = end_time - start_time

    cot = dataset1.shape[1]
    if cot <= 2:
        # gs = GridSpec(nrows=1, ncols=1)
        plt.figure(figsize=(20, 20))
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        colors = ['green', 'blue', 'red', 'black', 'yellow', 'violet']
        lb = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6']

        dataset2 = dataset1.values
        centers = np.array(centers)

        tam = pd.DataFrame(centers, columns=['Column1', 'Column2'])
        # mode='a' để ghi tiếp vào file.csv chứ không ghi đè lên nọi dung có săn trong file.csv
        tam.to_csv("C:\\GR1\\tam.csv", index=False, mode='w', header=False)

        nhan = pd.DataFrame(labels, columns=['Column1'])
        nhan.to_csv("C:\\GR1\\nhan.csv", index=False, mode='w')

        for i in range(2):

            fig = plt.subplots()
            # ax = plt.subplot()
            if i == 0:
                plt.scatter(dataset2[:, 0], dataset2[:, 1], s=50, alpha=0.5, color='red')
                plt.title('All points in original dataset')
                plt.savefig(r"C:\GR1\image\timge.png")
            else:
                for j in np.arange(k):
                    idx_j = np.where(np.array(labels) == j)[0]
                    plt.scatter(dataset2[idx_j, 0], dataset2[idx_j, 1], color=colors[j], label=lb[j], s=50, alpha=0.3, lw=0)
                    plt.scatter(centers[j, 0], centers[j, 1], marker='x', color=colors[j], s=100, label=lb[j])
                    plt.title(r'iteration')
                plt.savefig(r"C:\GR1\image\simge.png")

            # tinh chi so CHI

        mean_x = dataset1['Column1'].mean()
        mean_y = dataset1['Column2'].mean()

        # tinh BCSS
        BCSS = 0
        for i in np.arange(k):

            cumi = 0
            for j in labels:
                if j == i:
                    cumi = cumi + 1

            BCSS = BCSS + cumi * ((tam.iloc[i, 0] - mean_x) * (tam.iloc[i, 0] - mean_x) + (tam.iloc[i, 1] - mean_y) * (
                        tam.iloc[i, 1] - mean_y))
        # tinh WCSS
        WCSS = 0
        for i in np.arange(k):
            idx_i = np.where(np.array(labels) == i)[0]
            # tinh voi tung cum
            WCSS_i = 0
            for j in idx_i:
                WCSS_i = WCSS_i + (dataset2[j, 0] - centers[i, 0]) * (dataset2[j, 0] - centers[i, 0]) + (
                            dataset2[j, 1] - centers[i, 1]) * (dataset2[j, 1] - centers[i, 1])

            WCSS = WCSS + WCSS_i

        # CH
        CH = (BCSS / (k - 1)) / (WCSS / (len(dataset1) - k))

    else:
        centers = np.array(centers)

        tam = pd.DataFrame(centers, columns=['Column1', 'Column2', 'Column3', 'Column4'])
        # mode='a' để ghi tiếp vào file.csv chứ không ghi đè lên nọi dung có săn trong file.csv
        tam.to_csv("C:\\GR1\\tam.csv", index=False, mode='w', header=False)
        nhan = pd.DataFrame(labels, columns=['Column1'])
        nhan.to_csv("C:\\GR1\\nhan.csv", index=False, mode='w')

        # Tính trung bình của tất cả các cột
        means = dataset1.mean().values

        # Tính BCSS
        BCSS = 0
        for i in np.arange(k):
            cumi = 0

            for j in labels:
                if j == i:
                    cumi = cumi + 1

            BCSS += cumi * np.sum((centers[i] - means) ** 2)

        # Tính WCSS
        WCSS = 0
        for i in np.arange(k):
            idx_i = np.where(np.array(labels) == i)[0]
            if len(idx_i) == 0:
                continue
            WCSS_i = np.sum((dataset1.iloc[idx_i].values - centers[i]) ** 2)
            WCSS += WCSS_i

        # Tính chỉ số CH
        CH = (BCSS / (k - 1)) / (WCSS / (len(dataset1) - k))

    return execution_time, CH

def DBI(file_path):

    dataset1 = pd.read_csv(file_path)
    X = dataset1.values

    nhan = pd.read_csv("C:\\GR1\\nhan.csv")
    labels = nhan['Column1'].values
    n_clusters = len(np.unique(labels))
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    centroids = [np.mean(cluster, axis=0) for cluster in cluster_k]

    S = [np.mean(np.linalg.norm(cluster - centroid, axis=1)) for cluster, centroid in zip(cluster_k, centroids)]

    M = pairwise_distances(centroids, metric='euclidean')

    R = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):

            if i == j:
                R[i, j] = 0
            else:
                R[i, j] = (S[i] + S[j]) / M[i, j]

    D = []
    for i in range(n_clusters):
        D.append(np.max(R[i]))

    dbi = np.mean(D)
    return dbi

# truyen tham so
k= 3
m =1.7
epsilon = 0.00001
file_path = "C:\\GR1\\S_data.csv"
thoi_gian, CH = star(k, m, epsilon, file_path)
