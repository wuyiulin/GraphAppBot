import os
import multiprocessing
import time
import cv2
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy.stats import chisquare
import numpy as np
from scipy.fftpack import dct
import pdb

def timeViewer(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("This Process use " + str('{:.3f}'.format(end - start)) + "s")
        return result
    return inner

def OpenCVDCT(block, dict, N):
    block = cv2.dct(block)
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
                    # pdb.set_trace()
    return mask

def dotMask(block, mask, dict, N):
    DCT = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            # pdb.set_trace()
            DCT[i, j] = np.sum(np.dot(block, mask[i, j]))
            strDij = str(abs(DCT[i,j]))[0]
            
            if(strDij!='0'):
                dict[strDij]+=1
    # pdb.set_trace()


def MutiDCT(shared_data, DCTmask, block, lock, N):
    Sdata = np.frombuffer(shared_data.get_obj(), dtype=np.float32)
    localArr = np.zeros(9, dtype=np.float32)
    DCT = np.dot(DCTmask, block)

    for i in range(N):
        for j in range(N):
            strDij = str(abs(DCT[i,j]))[0]
            if(strDij!='0'):
                Dij = int(strDij)
                localArr[Dij-1]+=1

    with lock:
        Sdata += localArr

    
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
        oldDCT(block, dict, N)
        progress.update(1)
    
    # 計算結果
    result = 0
    SumValue = img.shape[0] * img.shape[1]
    # print("Single Dict:")
    # print(dict)
    # print("Single SumValue: "+ str(SumValue))
    for key in dict:
        dict[key] /= SumValue
        result += abs(BenfordsDict[key] - dict[key])
    
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
        OpenCVDCT(block, dict, N)
        progress.update(1)

    # 計算結果
    result = 0
    SumValue = img.shape[0] * img.shape[1]
    print("Single Dict:")
    print(dict)
    print("Single SumValue: "+ str(SumValue))
    for key in dict:
        dict[key] /= SumValue
        result += abs(BenfordsDict[key] - dict[key])
    
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
    print("Single Dict:")
    print(dict)
    print("Single SumValue: "+ str(SumValue))
    for key in dict:
        dict[key] /= SumValue
        result += abs(BenfordsDict[key] - dict[key])
    
    return result


@timeViewer
def MultiTransform(img, cores=None):
    # 如果沒有指定核心數，則查詢電腦可用的核心數並設定。
    if(cores==None):
        ratio = 0.8
        num_cores = int(multiprocessing.cpu_count()*ratio)

    # 設定區塊大小
    N = 8

    # 將圖像分成 8x8 的區塊
    blocks = [np.float32(img[i:i+N, j:j+N]-128) for i in range(0, img.shape[0], N) for j in range(0, img.shape[1], N)]

    # 宣告共享的記憶體
    shared_data = multiprocessing.Array('i', 9)
    for i in range(len(shared_data)):
        shared_data[i] = 0  # 初始化所有值為0

    # 初始化 Benford's Law 的標準 Array
    BenfordsArray = np.array([np.log10((i+1)/i) for i in range(1, 10)])

    # 多 process 處理 DCT 運算
    DCTmask = genrateDCTmask(N)
    lock = multiprocessing.Lock()
    processes = []
    num_processes = len(blocks)
    progress = tqdm(total=num_processes)
    for block in blocks:
        p = multiprocessing.Process(target=MutiDCT, args=(shared_data, DCTmask, block, lock, N))
        processes.append(p)
        p.start()
        progress.update(1)

    for p in processes:
        p.join()
    
    # 計算結果
    result = 0
    SumValue = np.sum(shared_data)
    result_data = shared_data / SumValue
    result_data = abs(result_data - BenfordsArray)
    result = np.sum(result_data)
    print("SumValue: " + str(SumValue))
    print(shared_data)

    return result




if __name__ == '__main__':

    image = cv2.imread('./Head.jpg', cv2.IMREAD_GRAYSCALE)
    # resultO = oldSingleTransform(image)
    resultS = OpenCVSingleTransform(image)
    resultF = SingleTransform(image)
    # resultM = MultiTransform(image)

    # print("這張照片的修圖程度(單執行緒)： " + str('{:.3f}'.format(resultO)) +"\n")
    print("這張照片的修圖程度(單執行緒)： " + str('{:.3f}'.format(resultS)) +"\n")
    print("這張照片的修圖程度(單執行緒)： " + str('{:.3f}'.format(resultF)) +"\n")
    # print("這張照片的修圖程度(多執行緒)： " + str('{:.3f}'.format(resultM)) +"\n")



