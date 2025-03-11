# 实现kmeans算法
import numpy as np

def distance(x, y):
    """
    计算两个点之间的距离
    """
    return sum((x-y)**2)**0.5  

def kmeans(data, k, max_iter=100):
    """
    kmeans算法
    """
    # 随机选择k个点作为初始聚类中心
    centers = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iter):
        # 计算每个点到每个聚类中心的距离
        distances = np.array([[distance(x, center) for center in centers] for x in data])
        # 根据距离最近的聚类中心，将每个点分配到对应的簇
        labels = np.argmin(distances, axis=1)
        # 计算每个簇的新聚类中心
        new_centers = np.array([data[labels==i].mean(axis=0) for i in range(k)])
        # 如果新的聚类中心和旧的聚类中心相同，则停止迭代
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return labels, centers

# 实现kmeans++算法
import numpy as np
def kmeans_plus(data, k, max_iter=100):
    """
    kmeans++算法
    """
    # 随机选择一个点作为第一个聚类中心
    centers = [data[np.random.choice(data.shape[0])]]
    for _ in range(1, k):
        # 计算每个点到最近的聚类中心的距离
        distances = np.array([min([distance(x, center) for center in centers]) for x in data])
        # 计算每个点被选择为下一个聚类中心的概率
        prob = distances**2 / sum(distances**2)
        # 根据概率选择下一个聚类中心
        centers.append(data[np.random.choice(data.shape[0], p=prob)])
    return kmeans(data, k, max_iter)

    
