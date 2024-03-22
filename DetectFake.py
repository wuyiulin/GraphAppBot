import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
import cv2
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy.stats import chisquare
import numpy as np
from scipy.fftpack import dct
from queue import Queue
import ctypes
import pdb

def timeViewer(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Function '{}' took {:.3f} seconds".format(func.__name__, end - start))
        return result
    return inner

def OpenCVDCT(block, dict, N):
    block = cv2.dct(block*100)
    for i in range(N):
        for j in range(N):
            strDij = str(abs(block[i,j]))[0]
            if(strDij!='0'):
                dict[strDij]+=1

def oldDCT(block, dict, N):
    para = 1/(2*np.sqrt(2*N))
    OrthogonalValue = 1/np.sqrt(2)
    for i in range(N):
        for j in range(N):
            if(i==0):
                Ci = OrthogonalValue
            else:
                Ci = 1
            if(j==0):
                Cj = OrthogonalValue
            else:
                Cj = 1

            Dij = 0
            for x in range(N):
                for y in range(N):
                    Dij += para * Ci * Cj * (block[x,y]) * np.cos((2*x+1)*i*np.pi / (2*N)) * np.cos((2*y+1)*j*np.pi / (2*N))
            strDij = str(abs(Dij))[0]
            if(strDij!='0'):
                dict[strDij]+=1

def genrateDCTmask(N):
    mask = np.zeros((N, N, N, N), dtype=np.float32)
    para = 1/(2*np.sqrt(2*N))
    OrthogonalValue = 1/np.sqrt(2)
    for i in range(N):
        for j in range(N):
            if(i==0):
                Ci = OrthogonalValue
            else:
                Ci = 1
            if(j==0):
                Cj = OrthogonalValue
            else:
                Cj = 1

            for x in range(N):
                for y in range(N):
                    mask[i, j, x, y] = (para * Ci * Cj * np.cos((2*x+1)*i*np.pi / (2*N)) * np.cos((2*y+1)*j*np.pi / (2*N)))
                    # print("Now i j x y:" + str(i) + " " + str(j) + " " + str(x) + " " + str(y) + " ")
    
    minimum = (np.min(abs(mask)))
    AMPfactor = 1
    
    while(minimum < 1):
        minimum *= 10
        AMPfactor *=10

    mask *= AMPfactor

    return mask

def dotMask(block, mask, dict, N):
    DCT = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            DCT[i, j] = np.sum(block * mask[i, j])
            strDij = str(abs(DCT[i,j]))[0]
            # print("i: " + str(i) + ", j: " + str(j))
            # pdb.set_trace()
            
            if(strDij!='0'):
                dict[strDij]+=1
    # pdb.set_trace()


def MutiDCT(args):
    mask, block, N = args
    localArr = np.zeros(9, dtype=np.float32)
    DCT = np.zeros((N,N), dtype=np.float32)
    
    for i in range(N):
        for j in range(N):
            DCT[i, j] = np.sum(block * mask[i, j])
            strDij = str(abs(DCT[i,j]))[0]
            if strDij != '0':
                Dij = int(strDij)
                localArr[Dij-1] += 1

    return localArr

    
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')



@timeViewer
def oldSingleTransform(img):
    # 初始化圖片的首位字典
    manager = multiprocessing.Manager()
    dict = manager.dict({str(i): 0 for i in range(1, 10)})

    # 初始化 Benford's Law 的標準字典
    BenfordsDict = {str(i): np.log10((i+1)/i)  for i in range(1, 10)}

    # 設定區塊大小
    N = 8

    # 將圖片分成可以被 8x8 整除的尺寸
    h, w = img.shape
    hPara = h // N
    wPara = w // N
    img = cv2.resize(img, (wPara*N, hPara*N))


    # 將圖片分成 8x8 的區塊
    blocks = [np.float32(img[i:i+N, j:j+N]-128) for i in range(0, img.shape[0], N) for j in range(0, img.shape[1], N)]

    # 算 DCT 並紀錄首位數字
    times = len(blocks)
    progress = tqdm(total=times)
    for block in blocks:
        oldDCT(block*1000, dict, N)
        progress.update(1)
    
    # 計算結果
    result = 0
    SumValue = img.shape[0] * img.shape[1]
    # print("Single Dict:")
    # print(dict)
    # print("Single SumValue: "+ str(SumValue))
    for key in dict:
        dict[key] /= SumValue
        result += abs(BenfordsDict[key] - dict[key]) * (1 / BenfordsDict[key])
    
    return result


@timeViewer
def OpenCVSingleTransform(img):
    # 初始化圖片的首位字典
    manager = multiprocessing.Manager()
    dict = manager.dict({str(i): 0 for i in range(1, 10)})

    # 初始化 Benford's Law 的標準字典
    BenfordsDict = {str(i): np.log10((i+1)/i)  for i in range(1, 10)}

    # 設定區塊大小
    N = 8

    # 將圖片分成可以被 8x8 整除的尺寸
    h, w = img.shape
    hPara = h // N
    wPara = w // N
    img = cv2.resize(img, (wPara*N, hPara*N))


    # 將圖片分成 8x8 的區塊
    blocks = [np.float32(img[i:i+N, j:j+N]-128) for i in range(0, img.shape[0], N) for j in range(0, img.shape[1], N)]

    # 算 DCT 並紀錄首位數字
    times = len(blocks)
    progress = tqdm(total=times)
    for block in blocks:
        OpenCVDCT(block*1000, dict, N)
        progress.update(1)

    # 計算結果
    result = 0
    SumValue = img.shape[0] * img.shape[1]
    # print("Single Dict:")
    # print(dict)
    # print("Single SumValue: "+ str(SumValue))
    for key in dict:
        dict[key] /= SumValue
        result += abs(BenfordsDict[key] - dict[key]) * (1 / BenfordsDict[key])
    # print("BenfordsDict: ")
    # print(BenfordsDict)
    # print("OurDict: ")
    # print(dict)

    return result


@timeViewer
def SingleTransform(img):
    # 初始化圖片的首位字典
    manager = multiprocessing.Manager()
    dict = manager.dict({str(i): 0 for i in range(1, 10)})

    # 初始化 Benford's Law 的標準字典
    BenfordsDict = {str(i): np.log10((i+1)/i)  for i in range(1, 10)}

    # 設定區塊大小
    N = 8

    # 初始化 DCTmask
    DCTmask = genrateDCTmask(N)

    # 將圖片分成可以被 8x8 整除的尺寸
    h, w = img.shape
    hPara = h // N
    wPara = w // N
    img = cv2.resize(img, (wPara*N, hPara*N))


    # 將圖片分成 8x8 的區塊
    blocks = [np.float32(img[i:i+N, j:j+N]-128) for i in range(0, img.shape[0], N) for j in range(0, img.shape[1], N)]

    # 算 DCT 並紀錄首位數字
    times = len(blocks)
    progress = tqdm(total=times)
    
    for block in blocks:
        # block = np.dot(block, DCTmask)
        # block = dct2(block)
        # pdb.set_trace()
        dotMask(block, DCTmask, dict, N)
        progress.update(1)

    # 計算結果
    result = 0
    SumValue = img.shape[0] * img.shape[1]
    # print("Single Dict:")
    # print(dict)
    # print("Single SumValue: "+ str(SumValue))
    # print("Single TotalNum: "+ str(TotalNum))
    for key in dict:
        dict[key] /= SumValue
        result += abs(BenfordsDict[key] - dict[key]) * (1 / BenfordsDict[key])
    # print("BenfordsDict: ")
    # print(BenfordsDict)
    # print("OurDict: ")
    # print(dict)
    
    return result


@timeViewer
def MultiTransform(img, cores=None):
    if cores is None:
        ratio = 0.8
        num_cores = int(multiprocessing.cpu_count() * ratio)

    N = 8
    blocks = [np.float32(img[i:i+N, j:j+N]-128) for i in range(0, img.shape[0], N) for j in range(0, img.shape[1], N)]

    BenfordsArray = np.array([np.log10((i+1)/i) for i in range(1, 10)])

    DCTmask = genrateDCTmask(N)
    progress = tqdm(total=len(blocks))
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(MutiDCT, ((DCTmask, block, N) for block in blocks))
        result_data = np.zeros(9)
        for localArr in results:
            result_data += localArr
            progress.update(1)

    SumValue = img.shape[0] * img.shape[1]
    SumValue_array = np.full(len(result_data), SumValue)
    result_data = result_data / SumValue_array
    for i in range(len(result_data)):
        result_data[i] = abs(result_data[i] - BenfordsArray[i]) * (1 / BenfordsArray[i])

    return np.sum(result_data)

@timeViewer
def SingleCTransform(imgPath):

    # Load C++ library
    lib = ctypes.cdll.LoadLibrary('./Detectlib/DetectFake.so')
    lib.DetectC.argtypes = [ctypes.c_char_p]
    lib.DetectC.restype = ctypes.c_float


    # Call C++ function
    c_imgPath = ctypes.c_char_p(imgPath.encode('utf-8'))

    result = lib.DetectC(c_imgPath)
    
    return result
    

if __name__ == '__main__':

    path = '/home/franky/Data/Lab/Project/SourceCode/GraphAppBot/Head.jpg'
    image = cv2.imread('./Head.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)

    resultO = oldSingleTransform(image)
    resultS = OpenCVSingleTransform(image)
    resultF = SingleTransform(image)
    resultM = MultiTransform(image)
    resultC = SingleCTransform(path)

    print("這張照片的修圖程度(單執行緒 土法煉鋼 Method)： " + str('{:.3f}'.format(resultO)) +"\n")
    print("這張照片的修圖程度(單執行緒 OpenCV  Method)： " + str('{:.3f}'.format(resultS)) +"\n")
    print("這張照片的修圖程度(單執行緒 Mask    Method)： " + str('{:.3f}'.format(resultF)) +"\n")
    print("這張照片的修圖程度(多執行緒 Mask    Method)： " + str('{:.3f}'.format(resultM)) +"\n")
    print("這張照片的修圖程度(單執行緒 CPP    Method)： " + str('{:.3f}'.format(resultC)) +"\n")


