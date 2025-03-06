import cv2

class DataAug():
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def rotate_image(self, angle, rotate_save_path):
        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
        
        # cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101, cv2.BORDER_WRAP
        rotated = cv2.warpAffine(self.image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

        cv2.imwrite(rotate_save_path, rotated)

    
    def flip_image(self, direction, flip_save_path):
        flipped = cv2.flip(self.image, direction) # horizontally: 1, vertically: 0, both: -1
        cv2.imwrite(flip_save_path, flipped)





if __name__ == "__main__":
    # image = cv2.imread('results/32-dims/rotation-test/UHD_g2.79e12_01.png')
    # rotated_image = rotate_image(image, 50)
    # cv2.imwrite('results/32-dims/rotation-test/UHD_g2.79e12_01_rotated.png', rotated_image)

    data_aug1 = DataAug('new-sparse/data-aug-test/noAGN_g7.55e11_06.png')
    data_aug1.flip_image(1, 'new-sparse/data-aug-test/noAGN_g7.55e11_06_H.png')