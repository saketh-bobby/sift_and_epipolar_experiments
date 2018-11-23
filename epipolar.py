import numpy as np
import cv2
import panorama_stitching
import matplotlib.pyplot as plt

# Reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html

def drawlines(img1,img2,lines,pts1,pts2):  
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

if __name__ == "__main__":
    UBIT = "sakethva"
    np.random.seed(sum([ord(c) for c in UBIT]))

    color_img1 = cv2.imread("./data/tsucuba_left.png")
    color_img2 = cv2.imread("./data/tsucuba_right.png")

    img1_gray = cv2.cvtColor(color_img1,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(color_img2,cv2.COLOR_BGR2GRAY)

    print("Task 2")

    print("2.1")

    kp_img1, kp_img2, des1, des2 = panorama_stitching.generate_keypoints(img1_gray, img2_gray)
    panorama_stitching.draw_keypoints(img1_gray, img2_gray, kp_img1, kp_img2, "task2")
    print("Generated keypoints and KNN matches")
    print("2.2")
    matches, matches_mask, good_matches, pts1, pts2 = panorama_stitching.get_knn_matches(kp_img1, kp_img2, des1, des2)

    draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matches_mask,
                flags = 0)

    knnMatches = cv2.drawMatchesKnn(color_img1, kp_img1, color_img2, kp_img2, matches, None, **draw_params)

    cv2.imwrite("task2_matches_knn.jpg", knnMatches)


    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

    print("F = ", F)

    print("2.3")

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    rand_pts1 = np.random.permutation(pts1)[:10]

    pts2 = pts2[mask.ravel()==1]
    rand_pts2 = np.random.permutation(pts2)[:10]

    # Find epilines from right image to left image
    lines1 = cv2.computeCorrespondEpilines(rand_pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    epi_left, _ = drawlines(img1_gray,img2_gray,lines1,rand_pts1,rand_pts2)
    
    # Find epilines from left image to right image
    lines2 = cv2.computeCorrespondEpilines(rand_pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    epi_right, _ = drawlines(img2_gray,img1_gray,lines2,rand_pts2,rand_pts1)
    
    cv2.imwrite("task2_epi_left.jpg", epi_left)
    cv2.imwrite("task2_epi_right.jpg", epi_right)

    print("Computed epipolar lines and plotted them")
    print("2.4")
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )
    
    
    image_left = cv2.imread("task2_epi_left.jpg")
    image_right = cv2.imread("task2_epi_right.jpg")
    

    disp_map = stereo.compute(image_left, image_right).astype(np.float32) / 16.0

    cv2.imwrite("task2_disparity.jpg", disp_map)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print("Disparity map has been generated and saved")