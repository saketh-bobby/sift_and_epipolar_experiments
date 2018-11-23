import numpy as np
import cv2


def generate_keypoints(img1, img2):

    sift = cv2.xfeatures2d.SIFT_create()

    kp_m1, des1 = sift.detectAndCompute(img1,None)
    kp_m2, des2 = sift.detectAndCompute(img2,None)

    return kp_m1, kp_m2, des1, des2

def draw_keypoints(color_img1, color_img2, kp_img1, kp_img2, task_name):
    kp1 = cv2.drawKeypoints(color_img1, kp_img1, None)
    kp2 = cv2.drawKeypoints(color_img2, kp_img2, None)

    cv2.imwrite("%s_sift1.jpg" % task_name, kp1)
    cv2.imwrite("%s_sift2.jpg" % task_name, kp2)


def get_knn_matches(kp1, kp2, des1, des2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    matchesMask = np.zeros(np.array(matches).shape)

    good_matches = []
    
    pts1 = []
    pts2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matchesMask[i]=[1,0]
            good_matches.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    return matches, matchesMask, good_matches, np.array(pts1), np.array(pts2)

def get_homography_matrix(kp_m1, kp_m2, good_matches):
    src_pts = np.float32([ kp_m1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_m2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    return H, mask


def get_10_random_inliers(color_img1, color_img2, kp1, kp2, good_matches, mask):
    matchesMask = np.array(mask.ravel().tolist())

    temp = np.where(matchesMask == 1)[0]
    perm_10 = np.random.permutation(temp)[:10]

    matchesMask = np.zeros(np.array(matchesMask).shape)
    np.put(matchesMask, perm_10, 1)

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img3 = cv2.drawMatches(mountain1, kp_m1, mountain2, kp_m2, good_matches,None,**draw_params)

    cv2.imwrite("task1_matches.jpg", img3)


def generate_panorama(img1, img2):

    kp_m1, kp_m2, des1, des2 = generate_keypoints(img1, img2)

    _, _, good_matches, _, _ = get_knn_matches(kp_m1, kp_m2, des1, des2)

    H, _ = get_homography_matrix(kp_m1, kp_m2, good_matches)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1_dims = np.float32([[[0, 0]], [[0, h1]], [[w1, h1]], [[w1, 0]]])

    img2_dims = np.float32([[[0, 0]], [[0, h2]], [[w2, h2]], [[w2, 0]]])

    # perspective transformation of image 1
    img1_dims_transformed = cv2.perspectiveTransform(img1_dims, H)
    
    pano = np.concatenate([img1_dims_transformed, img2_dims], axis=0)
    
    x_start, y_start = np.int32(pano.min(axis=0).ravel() - 0.5)
    x_end, y_end = np.int32(pano.max(axis=0).ravel() + 0.5)
    
    offset_x = -x_start
    offset_y = -y_start
    
    # transforming the homography matrix
    transform = np.array([[1,0, offset_x],[0,1, offset_y], [0,0,1]]) 

    H = np.dot(transform, H)

    # warp image 1
    pano_img = cv2.warpPerspective(img1, H, (x_end-x_start, y_end-y_start))
    
    # attaching img2 to the warped img1
    pano_img[offset_y:h1+offset_y,offset_x:w1+offset_x] = img2


    cv2.imwrite("task1_pano.jpg", pano_img)

if __name__ == "__main__":
    UBIT = "sakethva"
    np.random.seed(sum([ord(c) for c in UBIT]))

    mountain1 = cv2.imread("./data/mountain1.jpg")
    mountain2 = cv2.imread("./data/mountain2.jpg")

    m1_gray = cv2.cvtColor(mountain1,cv2.COLOR_BGR2GRAY)
    m2_gray = cv2.cvtColor(mountain2,cv2.COLOR_BGR2GRAY)

    print("Task 1")

    print("1.1")

    kp_m1, kp_m2, des1, des2 = generate_keypoints(m1_gray, m2_gray)
    draw_keypoints(mountain1, mountain2, kp_m1, kp_m2, "task1")

    print("1.2")
    matches, matches_mask, good_matches, pts1, pts2, = get_knn_matches(kp_m1, kp_m2, des1, des2)

    draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matches_mask,
                flags = 0)


    knnMatches = cv2.drawMatchesKnn(mountain1, kp_m1, mountain2, kp_m2, matches, None, **draw_params)

    cv2.imwrite("task1_matches_knn.jpg", knnMatches)

    print("1.3")
    
    H, mask = get_homography_matrix(kp_m1, kp_m2, good_matches)
    print("H = ", H)

    print(1.4)

    get_10_random_inliers(mountain1, mountain2, kp_m1, kp_m2, good_matches, mask)
    print("Generated 10 random inliers")
    print("1.5")

    generate_panorama(mountain1, mountain2)
    print("Generated panorama")
