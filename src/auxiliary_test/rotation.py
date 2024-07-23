import cv2


def rotate_image(image, angle):
    height, width = image.shape[:2]
    
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    
    # cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101, cv2.BORDER_WRAP
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    return rotated


if __name__ == "__main__":
    image = cv2.imread('results/32-dims/rotation-test/UHD_g2.79e12_01.png')
    rotated_image = rotate_image(image, 50)
    cv2.imwrite('results/32-dims/rotation-test/UHD_g2.79e12_01_rotated.png', rotated_image)