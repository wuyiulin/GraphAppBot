import os
import multiprocessing
import time
import cv2
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy.stats import chisquare
import numpy as np
import pdb

def timeViewer(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("This Process use " + str('{:.3f}'.format(end - start)) + "s")
        return result
    return inner

def DCT(block, dict, N):
    block = np.float32(block)
    dct = cv2.dct(block)
    for i in range(N):
        for j in range(N):
            strDij = str(abs(dct[i,j]))[0]
            if(strDij!='0'):
                dict[strDij]+=1
    # para = 1/(2*np.sqrt(2*N))
    # OrthogonalValue = 1/np.sqrt(2)
    # for i in range(N):
    #     for j in range(N):
    #         if(i==0):
    #             Ci = OrthogonalValue
    #         else:
    #             Ci = 1
    #         if(j==0):
    #             Cj = OrthogonalValue
    #         else:
    #             Cj = 1

    #         Dij = 0
    #         for x in range(N):
    #             for y in range(N):
    #                 Dij += para*Ci*Cj*(block[x,y]) * np.cos((2*x+1)*i*np.pi / (2*N)) * np.cos((2*y+1)*j*np.pi / (2*N))
    #         strDij = str(abs(Dij))[0]
    #         if(strDij!='0'):
    #             dict[strDij]+=1

@timeViewer
def SingleTransform(img):

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
        DCT(block, dict, N)
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

    # 初始化圖片的首位字典
    manager = multiprocessing.Manager()
    dict = manager.dict({str(i): 0 for i in range(1, 10)})

    # 初始化 Benford's Law 的標準字典
    BenfordsDict = {str(i): np.log10((i+1)/i)  for i in range(1, 10)}

    ## 算 DCT 並紀錄首位數字
    times = len(blocks)
    progress = tqdm(total=times)

    # 建立多進程的進程池
    pool = multiprocessing.Pool(processes=num_cores)

    # 對每個 block 啟動一個進程處理
    for block in blocks:
        pool.apply_async(DCT, args=(block, dict, N))
        progress.update(1)
    
    # 等待所有進程完成
    pool.close()
    pool.join()
    
    # 計算結果
    result = 0
    SumValue = sum(dict.values())
    for key in dict:
        dict[key] /= SumValue
        result += abs(BenfordsDict[key] - dict[key])
    print("SumValue: " + str(SumValue))
    print(dict)

    # # 將字典的值轉換為列表
    # observed = list(dict.values())
    # expected = list(BenfordsDict.values())

    # # 將列表轉換為 numpy array
    # observed_array = np.array(observed)
    # expected_array = np.array(expected)
    # # 進行卡方檢定
    # chi2_stat, p_val = chisquare(observed_array, f_exp=expected_array)
    # print("卡方檢定值: " + str('{:.3f}'.format(chi2_stat)) + ", P值: " + str('{:.3f}'.format(p_val)))
    return result




if __name__ == '__main__':

    image = cv2.imread('./Head.jpg', cv2.IMREAD_GRAYSCALE)
    # resultS = SingleTransform(image)
    resultM = MultiTransform(image)

    # print("這張照片的修圖程度(單執行緒)： " + str('{:.3f}'.format(resultS)) +"\n")
    print("這張照片的修圖程度(多執行緒)： " + str('{:.3f}'.format(resultM)) +"\n")



