import cv2
mode = 1


def gaussianBlur(main, value):
    main.new_image = main.image
    if value == 0:
        value = 1;
    if value % 2 == 0:
        value += 1
    kernel_size = (value, value)  # +1 is to avoid 0
    main.new_image = cv2.GaussianBlur(main.new_image, kernel_size, 0)
    main.updateImage()
# def checkboxEvt(self, state):
#     if (QtCore.Qt.Checked == state):
#         self.new_image = self.image
#         self.new_image = cv2.cvtColor(self.new_image, cv2.COLOR_BGR2GRAY)
#
#         self.updateImage()

# def mediumBlur(self, value):
#     cv2.blu

def mediumBlur(main, value):
    main.new_image = main.image
    if value == 0:
        value = 1;
    if value % 2 == 0:
        value += 1
    kernel_size = (value, value)  # +1 is to avoid 0
    main.new_image = cv2.boxFilter(main.new_image, (value, value))
    main.updateImage()

# def threshsold(main, value):
