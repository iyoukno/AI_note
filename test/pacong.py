'''
@Project ：test 
@File    ：pacong.py
@Author  ：yuk
@Date    ：2024/5/16 18:04 
description：爬虫1 只能爬30张
'''

import requests # 爬虫必备
import time # 限制爬虫速度
import os # 新建指定存储文件夹


import requests
import time
import random


def get_ip():
    url = "这里放你自己的API链接"
    while 1:
        try:
            r = requests.get(url, timeout=10)
        except:
            continue

        ip = r.text.strip()
        if '请求过于频繁' in ip:
            print('IP请求频繁')
            time.sleep(1)
            continue
        break
    proxies = {
        'https': '%s' % ip
    }

    return proxies

def get_img_url(keyword,page_num):
    """发送请求，获取接口中的数据"""
    # 接口链接
    url = 'https://image.baidu.com/search/acjson?'
    # 请求头模拟浏览器
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
    # 构造网页的params表单
    params = {
        'tn': 'resultjson_com',
        'logid': '8335606437438811132',
        'ipn': 'rj',
        'ct': '201326592',
        'is': '',
        'fp': 'result',
        'queryWord': f'{keyword}',
        'word': f'{keyword}',
        'cl': '1018',
        'lm': '',
        'ie': 'utf-8',
        'oe': 'utf-8',
        'adpicid': '',
        'st': '-1',
        'z': '',
        'ic': '',
        'hd': '',
        'latest': '',
        'copyright': '',
        's': '',
        'se': '',
        'tab': '',
        'width': '',
        'height': '',
        'face': '0',
        'istype': '2',
        'qc': '',
        'nc': '1',
        'fr': '',
        'expermode': '',
        'force': '',
        'cg': '',
        'pn': page_num*30,
        'rn': 30,
        'gsm': '5a',
    }
    # 携带请求头和params表达发送请求
    response  = requests.get(url=url, headers=headers, params=params)
    # 设置编码格式
    response.encoding = 'utf-8'
    # 转换为json
    json_dict = response.json()
    # 定位到30个图片上一层
    data_list = json_dict['data']
    # 删除列表中最后一个空值
    del data_list[-1]
    # 用于存储图片链接的列表
    img_url_list = []
    for i in data_list:
        img_url = i['thumbURL']
        # 打印一下图片链接
        # print(img_url)
        img_url_list.append(img_url)
    # 返回图片列表
    return img_url_list


def get_down_img(img_url_list, i, key):
    # 在当前路径下生成存储图片的文件夹
    if not os.path.exists(r"E:\pacong\img\fire_negative"):
        os.mkdir(r"E:\pacong\img\fire_negative")
    # 定义图片编号
    n = 0
    for img_url in img_url_list:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'}
        # 调用get_ip函数，获取代理IP
        # proxies = get_ip()
        # 每次发送请求换代理IP，获取图片，防止被封
        img_data = requests.get(url=img_url, headers=headers).content
        # 拼接图片存放地址和名字
        img_path = 'E:/pacong/img/fire_negative/' + key+'_'+ str(i)+"_"+str(n) + '.jpg'
        # 将图片写入指定位置
        with open(img_path, 'wb') as f:
            f.write(img_data)
            print('E:/pacong/img/fire_negative/' + str(n) + '.jpg 下载完成')
        # 图片编号递增
        n = n + 1



if __name__ == '__main__':

    keyword = '夜晚堵车'
    # 2. 获取指定关键词的图片链接

    # 3. 下载图片到指定位置
    for i in range(1,10):
        img_url_list = get_img_url(keyword, i)
        get_down_img(img_url_list,i,keyword)
