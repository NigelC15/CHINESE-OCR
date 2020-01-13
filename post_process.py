# -*- coding: utf-8 -*-

import re
import heapq
from enum import Enum
import Levenshtein
import pandas as pd
from glob import glob
import numpy as np
from PIL import Image
import time
import jieba
import jieba.posseg as pseg


def get_ocr_result(model, path='./img_for_demo/healthrecord_original.jpg'):
    
    im = Image.open(path)
    img = np.array(im.convert('RGB'))
    t = time.time()
    result,img = model.model(img,model='crnn') # if model == crnn ,you should install pytorch
    print("It takes time:{}s".format(time.time()-t))
    
    return result, img

def str_similarity(str_1, str_2, num_alpha_chn_filter=True):
    
    
    
    if len(str_1) > len(str_2):
        shorter = str_2
        longer = str_1
    else:
        shorter = str_1
        longer = str_2
        
    if num_alpha_chn_filter: 
        re_pattern = re.compile(r'[^A-Za-z0-9\u4e00-\u9fa5]')
        shorter = re.sub(re_pattern, '', shorter)
        longer = re.sub(re_pattern, '', longer)
    
    if len(shorter) < 3:
        return 100
    
    start_ind = 0
    end_ind = len(longer) - len(shorter)
    
    if start_ind == end_ind:
        return Levenshtein.distance(longer, shorter) / len(shorter)
    
    similarity = 1
    
    for ind in range(start_ind, end_ind + 1):
        s = Levenshtein.distance(longer[ind: (ind+len(shorter))], shorter) / len(shorter)
        if s < similarity:
            similarity = s
#     print(similarity, shorter, longer)
    return similarity
#    return Levenshtein.distance(str_1.upper(), str_2.upper()) / len(str_2)
#    return Levenshtein.ratio(str_1, str_2)


def ocr_result_to_df(ocr_result, title_path='./med_dict/title_name.csv', item_name_path='./med_dict/item_name.csv'):
    # find title
    df = pd.read_csv(title_path)

    title_key = -1
    flag = False
    for key in ocr_result:
        for ind in range(len(df['col_names'])):
            if str_similarity(ocr_result[key][1], df['col_names'][ind]) < 0.33:
                flag = True
                title_key = key
                ocr_result[key][1] = df['col_names'][ind]
                break
            else: continue        
        if flag: 
            break 
    
    tmp_dict = {}
    for key in ocr_result:
        if abs(ocr_result[key][0][1] - ocr_result[title_key][0][1]) <= 10:
            tmp_dict[key] = ocr_result[key]
    #根据result_copy[key][0][0]排序
    tmp_dict = dict(sorted(tmp_dict.items(), key=lambda x:x[1][0][0]))
    
    col_name = []
    for key in tmp_dict:
        col_name.append(tmp_dict[key][1])
    df_output = pd.DataFrame(columns=col_name)
    
    # find item
    df_item = pd.read_csv(item_name_path)

    item_keys = []
    flag = False
    for ind in range(min(tmp_dict), len(ocr_result)):
        if ind in tmp_dict.keys():
            continue
        else:
            for jnd in range(len(df_item['item_names'])):
                if str_similarity(ocr_result[ind][1], df_item['item_names'][jnd]) < 0.33:
                    if ind not in item_keys:
                        item_keys.append(ind)
                        ocr_result[ind][1] = df_item['item_names'][jnd]
                    
    # item_keys
    for item_key in item_keys:
        tmp_dict = {}
        for key in ocr_result:
            if abs(ocr_result[key][0][1] - ocr_result[item_key][0][1]) <= 10:
                tmp_dict[key] = ocr_result[key]
        #根据result_copy[key][0][0]排序
        tmp_dict = dict(sorted(tmp_dict.items(), key=lambda x:x[1][0][0]))

        tmp = []
        for key in tmp_dict:
            tmp.append(tmp_dict[key][1])
        while len(tmp) < len(col_name):
            tmp.append(' ')
        df_output.loc[df_output.shape[0]] = tmp
    return df_output


def get_basic_info_df(ocr_result, hospital_path='./med_dict/hospital_name.csv', surname_path='./med_dict/surname.csv'):
    
    hospital = None
    doctor = None
    name = None
    name_flag = False
    sex = None
    sex_str = ['男', '女', 'male', 'female']
    date = None
    diagnosis = None
    df_title = pd.read_csv('./med_dict/title_name.csv')
    item_flag = False
    total = None
    paid = None
    
    df_hospital = pd.read_csv(hospital_path)
    
    for key in ocr_result:
        
        # name
        if name_flag:
            continue
        # limits the length of name string
        re_pattern = re.compile(r'[^\u4e00-\u9fa5]')
        tmp_str = re.sub(re_pattern, '', ocr_result[key][1])
        if len(tmp_str) > 5:
            continue
        
        for ind in range(len(df_title['col_names'])):
            if str_similarity(ocr_result[key][1], df_title['col_names'][ind]) < 0.33:
                item_flag = True
                break
        if item_flag: break
        
        words = pseg.cut(ocr_result[key][1])
#         print('===============================================')
        eng_flag = 0
        nr_flag = 0
        for word, flag in words:
# #             print('%s----------%s' % (word, flag))
#             print(ocr_result[key][1])
            if flag == 'eng':
                eng_flag = 1
            if flag == 'nr':
                nr_flag = 1
            
            if flag == 'v' or flag == 'c' or flag == 'q':
#                 print('xxxxxxxxxxxxxxx')
                nr_flag = 0
                name_flag = False
                name = None
                break

            if eng_flag and nr_flag and 'dr' not in ocr_result[key][1][:3].lower():
                name = ocr_result[key][1]
                name_flag = True
#                print(ocr_result[key])
                break
    
            if not eng_flag and nr_flag:
                name = ocr_result[key][1]
                name_flag = True
                continue
    
    
    for key in ocr_result:
        
        #diagnosis
        if str('diagnosis') in ocr_result[key][1].lower()[:10]:
#            print(key, ocr_result[key][1])
            if len(ocr_result[key][1]) < 7:
                tmp_dict = {}
                for key_tmp in ocr_result:
                    if abs(ocr_result[key][0][1] - ocr_result[key_tmp][0][1]) <= 9:
                        tmp_dict[key_tmp] = ocr_result[key_tmp]
                tmp_list = list(tmp_dict.values())
                for item in tmp_list:
                    if 'date' not in item[1].lower():
                        date = item[1]
            else:
                posi = ocr_result[key][1].lower().find('diagnosis:')
                posi = posi + 10
                diagnosis = ocr_result[key][1][posi:]
            continue
        
        #sex
        if ocr_result[key][1] in sex_str:
            sex = ocr_result[key][1]
            continue
        #dr
        if len(ocr_result[key][1]) > 4:
            if str('dr') in ocr_result[key][1].lower()[:4]:
                posi = ocr_result[key][1].lower().find('dr')
                posi = posi + 2
                doctor = ocr_result[key][1][posi:]
            elif str('dr.') in ocr_result[key][1].lower()[:4]:
                posi = ocr_result[key][1].lower().find('dr.')
                posi = posi + 3
                doctor = ocr_result[key][1][posi:]
        #date
        if str('date:') in ocr_result[key][1].lower():
#            print(key, ocr_result[key][1])
            if len(ocr_result[key][1]) < 7:
                tmp_dict = {}
                for key_tmp in ocr_result:
                    if abs(ocr_result[key][0][1] - ocr_result[key_tmp][0][1]) <= 9:
                        tmp_dict[key_tmp] = ocr_result[key_tmp]
                tmp_list = list(tmp_dict.values())
                for item in tmp_list:
                    if 'date' not in item[1].lower():
                        date = item[1]
            else:
                posi = ocr_result[key][1].lower().find('date:')
                posi = posi + 5
                date = ocr_result[key][1][posi:]
            continue
        #hospital
        for ind in range(len(df_hospital['hospital_names'])):
            if str_similarity(ocr_result[key][1], df_hospital['hospital_names'][ind]) < 0.33:
                ocr_result[key][1] = df_hospital['hospital_names'][ind]
                hospital = df_hospital['hospital_names'][ind]
                break
            else: continue
        
        #total
        if len(ocr_result[key][1]) > 8:
            continue
        if str_similarity('total:', ocr_result[key][1]) < 0.33:
            tmp_dict = {}
            for key_tmp in ocr_result:
                if abs(ocr_result[key][0][1] - ocr_result[key_tmp][0][1]) <= 9:
                    tmp_dict[key_tmp] = ocr_result[key_tmp]
            tmp_dict = dict(sorted(tmp_dict.items(), key=lambda x:x[1][0][0]))
            tmp_list = list(tmp_dict.values())
            for ind in range(len(tmp_list)):
                if str_similarity('total:', tmp_list[ind][1]) < 0.33:
                    total = tmp_list[ind+1][1]
                    break
            continue
               
        #paid
        if len(ocr_result[key][1]) > 6:
            continue
        if str_similarity('paid:', ocr_result[key][1]) < 0.33:
            tmp_dict = {}
            for key_tmp in ocr_result:
                if abs(ocr_result[key][0][1] - ocr_result[key_tmp][0][1]) <= 9:
                    tmp_dict[key_tmp] = ocr_result[key_tmp]
            tmp_dict = dict(sorted(tmp_dict.items(), key=lambda x:x[1][0][0]))
            tmp_list = list(tmp_dict.values())
            for ind in range(len(tmp_list)):
                if str_similarity('paid:', tmp_list[ind][1]) < 0.33:
                    paid = tmp_list[ind+1][1]
                    break
            continue
        
    
    return {'hospital':hospital, 'doctor':doctor, 'name':name, 'sex':sex,\
            'date':date, 'diagnosis':diagnosis, 'total':total, 'paid':paid}



IMG_TYPE = Enum('IMG_TYPE', ('invoice', 'receipt', 'medicalreport', 'others'))
IMG_TYPE_KEYWORD = {'invoice': ['invoice'], 
                    'receipt': ['transactioncode', 'receipt', 'diagnosis'], 
                    'medicalreport': ['参考值', 'range', 'sex']
                   }


def longest_common_subsequence(A, B):
    '''get the longest common subsequence between str1 and str2
    
    Args:
        A: string
        B: string
        
    Returns:
        result: str, the longest common subsequence string
    
    '''
    n = len(A)
    m = len(B)

    opt = [[0 for i in range(0,m+1)] for j in range(0,n+1)]

    pi = [[0 for i in range(0,m+1)] for j in range(0,n+1)]

    for i in range(1,n+1):
        for j in range(1,m+1):
            if A[i-1] == B[j-1]:
                opt[i][j] = opt[i-1][j-1] + 1
                pi[i][j] = 0
            elif opt[i][j-1] >= opt[i-1][j]:
                opt[i][j] = opt[i][j-1]
                pi[i][j] = 1
            else:
                opt[i][j] = opt[i-1][j]
                pi[i][j] = 2
    i = n
    j = m
    S = ''

    while i>0 and j>0:
        if pi[i][j] == 0:
            S = A[i-1] + S
            i-=1
            j-=1
        elif pi[i][j] == 2:
            i-=1
        else:
            j-=1
    return S


def match_img_keyword(target_text, keyword, threshold=0.3):
    '''match the keyword in one img text info line
    
    Args:
        target_text: string
        keyword: string
        
    Returns:
        result: float
    '''
    filter_text = longest_common_subsequence(target_text, keyword)
    ratio = str_similarity(filter_text, keyword, num_alpha_chn_filter=False)
    result = ratio < threshold
    return result


def filter_keyword(target_word, keyword):
    result = keyword in target_word
    return result


def classify_text(raw_text, threshold=0.3):
    ''' classifiy with the raw text
    
    Args:
        raw_text: dict, raw text infos from img detecting
        threshold: float, default 0.3, threshold ratio in the function of cal_levenshtein_ratio.  
        
    Returns:
        result: tunple, (string, int), 
            (invoice, 1)
            (receipt, 2) 
            (medical_report: 3)
            (others, 0) 
        
    '''
    result = None
    cal_result_list = []
    all_img_type = IMG_TYPE.invoice, IMG_TYPE.receipt, IMG_TYPE.medicalreport
    re_pattern = re.compile(r'[^A-Za-z\u4e00-\u9fa5]')
    try:
        for _, img_type in enumerate(all_img_type):
            for key, _ in enumerate(raw_text):
                str1 = raw_text[key][1]
#                print("=============")
#                print(f"old str1 {str1}")
                str1 = ''.join(re_pattern.sub(' ', str1))
                str1 = str1.lower()
#                print(f"new str1 {str1}")
                if len(str1) == 0:
                    continue
                str2 = img_type.name
                if str2 in str1:
                    img_distance = (img_type.name, 0)
                    heapq.heappush(cal_result_list, img_distance)
                    break 
                if len(list(filter(lambda n: match_img_keyword(str1, n, threshold=0.1), IMG_TYPE_KEYWORD[img_type.name]))) > 0:
                    img_distance = (img_type.name, 0)
                    heapq.heappush(cal_result_list, img_distance)
                    break
                ratio = str_similarity(str1, str2, num_alpha_chn_filter=False)
                if ratio < threshold:
                    img_distance = (img_type.name, ratio)
                    heapq.heappush(cal_result_list, img_distance)

    except Exception as e:
        raise e
#     print(f"cal_result_list: {cal_result_list}")
    result = heapq.nsmallest(1, cal_result_list, key=lambda x:x[0])
    if len(result) == 0:
        result = [('others', 0)]
#     print(f"result: {result}")
    return result