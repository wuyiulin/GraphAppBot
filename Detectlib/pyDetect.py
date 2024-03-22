import ctypes

path = "/home/franky/Data/Project/空車格分類/msn-main/image/ILSVRC2012_img_val/ILSVRC2012_val_00000014.JPEG"


# Load C++ library
lib = ctypes.cdll.LoadLibrary('./DetectFake.so')

# Call C++ function
c_imgPath = ctypes.c_char_p(path.encode('utf-8'))
result = lib.DetectC(c_imgPath)