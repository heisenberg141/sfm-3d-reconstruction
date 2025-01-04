import numpy as np
import cv2

from EstimateFundamentalMatrix import *

image1_points = np.array([  [  7.15528, 197.921],
                            [ 17.9628,  246.402],
                            [ 21.5457,  330.566],
                            [ 48.7564,  33.4742],
                            [ 49.7568,   54.0226 ],
                            [ 55.2032,  489.003  ],
                            [ 57.4337,  594.943  ],
                            [ 58.9126,  200.007  ]])
image2_points = np.array([  [ 11.255,  225.237 ],
                            [359.959,  188.997 ],
                            [527.825,  443.917 ],
                            [611.037,  217.304 ],
                            [211.706,  547.995 ],
                            [240.859,  445.121 ],
                            [778.089, 574.977 ],
                            [ 89.0649, 228.316 ]])

# print(EstimateFundamentalMatrix(image1_points, image2_points))
F_est = EstimateFundamentalMatrix(image1_points, image2_points)
u,s,vT = np.linalg.svd(F_est)
print(f"F est: \n",F_est)
F_gt = cv2.findFundamentalMat(image1_points,image2_points,cv2.FM_8POINT)[0]
u,s,vT = np.linalg.svd(F_gt)
print(f"F GT: \n", F_gt)
print(f"MSE: \n{np.mean((F_gt-F_est)**2)}")

