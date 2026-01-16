import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Images ---
# Load the source image (the one to be warped)
img_src = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
# Load the destination image (the one to align to)
img_dst = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

if img_src is None or img_dst is None:
    print("Error: Could not load images. Check the file paths.")
else:
    # --- 2. Find Keypoints and Descriptors ---
    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Find keypoints and descriptors for both images
    kp1, des1 = orb.detectAndCompute(img_src, None)
    kp2, des2 = orb.detectAndCompute(img_dst, None)

    # --- 3. Match Features ---
    # Create a Brute-Force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only the best 50 matches for clarity
    good_matches = matches[:50]

    # --- 4. Find Homography ---
    # Extract location of good matches
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

    # Find the homography matrix using RANSAC
    # M is the 3x3 homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("Computed Homography Matrix:\n", M)

    # --- 5. Warp Source Image ---
    # Get the dimensions of the destination image
    h, w = img_dst.shape
    
    # Warp the source image to align with the destination image's perspective
    img_warped = cv2.warpPerspective(img_src, M, (w, h))

    # --- 6. Display Results ---
    # Convert grayscale images to BGR for displaying with Matplotlib

    img_src = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
    img_dst = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    img_warped = cv2.warpPerspective(img_src, M, (w, h))
    img_src_bgr = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR)
    img_dst_bgr = cv2.cvtColor(img_dst, cv2.COLOR_GRAY2BGR)
    img_warped_bgr = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2BGR)
    
    # Draw the matches for visualization
    img_matches = cv2.drawMatches(img_src, kp1, img_dst, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(16, 8))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matches')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_dst_bgr, cv2.COLOR_BGR2RGB))
    plt.imsave('1_.jpg', cv2.cvtColor(img_dst_bgr, cv2.COLOR_BGR2RGB))
    plt.title('Destination Image')
    plt.axis('off')


    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_warped_bgr, cv2.COLOR_BGR2RGB))
    plt.imsave('2_.jpg', cv2.cvtColor(img_warped_bgr, cv2.COLOR_BGR2RGB))
    plt.title('Warped Source Image')
    plt.axis('off')

    plt.suptitle('Homography Estimation using ORB', fontsize=16)
    plt.show()