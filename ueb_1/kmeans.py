import numpy as np
import cv2
from typing import Tuple

############################################################
#
#                       KMEANS
#
############################################################

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the overall distance or cluster centers positions




def initialize_clusters(img: np.ndarray, num_clusters: int, custom_init: bool = False) -> np.ndarray:
    """
    Initialize cluster centers by randomly selecting pixels from the image.
    
    :param img (np.ndarray): The image array.
    :param num_clusters (int): The number of clusters to initialize.
    :return np.ndarray: Array of initial cluster centers.
    """
    
    # YOUR CODE HERE  
    # HINT: you load your images in uint8 format. convert your initial centers to float32 -> initial_centers.astype(np.float32)

    ## NOTE !!!!!!!!!
    ## To get full points you - ADDITIONALLY - have to develop your own init method. Please read the assignment!
    ## It should work with both init methods.

    def _init_kmeanspp(pixels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """k‑means++ initialisation (optional bonus)."""
        N = pixels.shape[0]
        centers = np.empty((k, 3), dtype=np.float32)

        centers[0] = pixels[rng.integers(N)]

        # Squared distances to the closest centre so far
        d2 = np.linalg.norm(pixels - centers[0], axis=1) ** 2

        for i in range(1, k):
            probs = d2 / d2.sum()
            centers[i] = pixels[rng.choice(N, p=probs)]
            new_d2 = np.linalg.norm(pixels - centers[i], axis=1) ** 2
            d2 = np.minimum(d2, new_d2)

        return centers


    pixels = img.reshape(-1, 3)
    rng = np.random.default_rng(42)  # reproducible marking

    if not custom_init:
        idx = rng.choice(pixels.shape[0], num_clusters, replace=False)
        return pixels[idx].astype(np.float32)
    else:
        return _init_kmeanspp(pixels, num_clusters, rng)


def assign_clusters(img: np.ndarray, cluster_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Assign each pixel in the image to the nearest cluster center and calculate the overall distance.

    :param img (np.ndarray): The image array.
    :param cluster_centers (np.ndarray): Current cluster centers.   
    :return Tuple[np.ndarray, np.ndarray, float]: Tuple of the updated image, cluster mask, and overall distance.
    """
  
    # YOUR CODE HERE  
    # HINT: 
    # 1. compute distances per pixel
    # 2. find closest cluster center for each pixel
    # 3. based on new cluster centers for each pixel, create new image with updated colors (updated_img)
    # 4. compute overall distance just to print it in each step and see that we minimize here
    # you return updated_img.astype(np.uint8), closest_clusters, overall_distance
    # the updated_img is converted back to uint8 just for display reasons

    h, w, _ = img.shape
    pixels = img.reshape(-1, 3).astype(np.float32)  # (N,3)
    centers = cluster_centers[None, :, :]  # (1,k,3)

    # distance of all pixels to cluster centers
    dists = np.linalg.norm(pixels[:, None, :] - centers, axis=2)
    labels = dists.argmin(axis=1)  # (N,)

    # quantised pixels and overall error (sum‑squared‑error)
    quantised = cluster_centers[labels].reshape(h, w, 3)
    sse = np.sum((pixels - cluster_centers[labels]) ** 2)

    return quantised.astype(np.uint8), labels, float(sse)


def update_cluster_centers(img: np.ndarray, cluster_assignments: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Update cluster centers as the mean of assigned pixels.

    :param img (np.ndarray): The image array.
    :param cluster_assignments (np.ndarray): Cluster assignments for each pixel.
    :param num_clusters (int): Number of clusters.
    :return np.ndarray: Updated cluster centers.
    """
    
    # YOUR CODE HERE  
    # HINT: Find the new mean for each center and return new_centers (those are new RGB colors)

    pixels = img.reshape(-1, 3).astype(np.float32)
    new_centers = np.empty((num_clusters, 3), dtype=np.float32)
    rng = np.random.default_rng()

    for k in range(num_clusters):
        members = pixels[cluster_assignments == k]
        if members.size:  # non‑empty cluster
            new_centers[k] = members.mean(axis=0)
        else:  # empty → random pixel
            new_centers[k] = pixels[rng.integers(pixels.shape[0])]

    return new_centers


def kmeans_clustering(img: np.ndarray, num_clusters: int = 3, max_iterations: int = 100, tolerance: float = 0.01,
                      custom_init:bool = False) -> np.ndarray:
    """
    Apply K-means clustering to do color quantization. Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges, centers don't change in between iterations anymore. 
    
    :param img (np.ndarray): The image to be segmented.
    :param num_clusters (int): The number of clusters.
    :param max_iterations (int): The maximum number of iterations.
    :param tolerance (float): The convergence tolerance.
    :return np.ndarray: The segmented image.
    """
    
    # YOUR CODE HERE  
    # initialize the clusters
    # for loop over max_iterations
    # in each loop
    # 1. assign clusters, this gives you a quantized image
    # 2. update cluster centers
    # 3. check for early break with tolerance
    # return updated_img

    centers = initialize_clusters(img, num_clusters, custom_init=custom_init)

    prev_error = np.inf
    for it in range(max_iterations):
        quant_img, labels, error = assign_clusters(img, centers)
        centers_new = update_cluster_centers(img, labels, num_clusters)

        # % change in SSE
        rel_change = abs(prev_error - error) / (prev_error + 1e-9)
        print(f"Iter {it:2d}: SSE = {error:,.0f},  Delta = {rel_change * 100:5.2f}%")

        if rel_change < tolerance:
            break

        centers, prev_error = centers_new, error

    return quant_img


def load_and_process_image(file_path: str, scaling_factor: float = 0.5) -> np.ndarray:
    """
    Load and preprocess an image.
    
    :param file_path (str): Path to the image file.
    :param scaling_factor (float): Scaling factor to resize the image.        
    :return np.ndarray: The preprocessed image.
    """
    image = cv2.imread(file_path)

    # Note: the scaling helps to do faster computation :) 
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image

def main():
    file_path = './graffiti.png'
    num_clusters = 8
    
    img = load_and_process_image(file_path)
    segmented_img = kmeans_clustering(img, num_clusters, custom_init=True)
    
    cv2.imshow("Original", img)
    cv2.imshow("Color-based Segmentation Kmeans-Clustering", segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
