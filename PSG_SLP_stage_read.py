#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
# @author : biao chen
# @Email : chenbiao@sleepace.net
# @Project : Python_Files
# @File : PSG_time_cal.py
# @Software: PyCharm
# @Time : 2021/5/20 下午6:07
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏ ┓    ┏┓
            ┏ ┛┻ ━━━┛┻┓
            ┃     ☃   ┃
            ┃  ┳┛  ┗┳  ┃
            ┃     ┻    ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃  神兽保佑 ┣┓
              ┃　永无BUG！┏┛
              ┗┓┓┏━┳┓┏┛
               ┃┫┫  ┃┫┫
               ┗┻┛  ┗┻┛
"""

# import logging
# import logging.config
#
# # logging.config.fileConfig('logconfig.ini')

from datetime import timedelta
from slp_utils.utils import *

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")



par_dir = Path.cwd()
sub_dir = 'dcol'
slp_raw = 'slp_raw_csv'
slp_hr = 'slp_hr'

psg_raw_pr = 'psg_pr_csv'
psg_pr = 'psg_batch_PR'
psg_rr = 'psg_batch_RR'
psg_pr_rr = 'psg_pr_rr'
slp_bulk = 'slp_batch_output'
psg_bulk = 'psg_batch_output'
slp_dir = 'slp_hr_br'
psg_dir = 'psg_pr_rr'
img_dir = 'img'
txt_dir = 'txt_record'


cur_dir = par_dir.joinpath(sub_dir)
sav_slp_raw = path_concat(cur_dir,slp_raw)
sav_slp_hr = path_concat(cur_dir,slp_hr)
sav_psg_raw = path_concat(cur_dir,psg_raw_pr)
sav_psg_pr = path_concat(cur_dir,psg_pr)
sav_psg_rr = path_concat(cur_dir,psg_rr)
sav_psg_pr_rr = path_concat(cur_dir,psg_pr_rr)


cat_slp = path_concat(cur_dir,slp_bulk)
cat_psg = path_concat(cur_dir,psg_bulk)
src_slp = path_concat(cur_dir,slp_dir)
src_psg = path_concat(cur_dir,psg_dir)
src_img = path_concat(cur_dir,img_dir)
src_txt = path_concat(cur_dir,txt_dir)



## PSG呼吸率为空
if not os.listdir(cat_psg):

    pr_fileList = os.listdir(sav_psg_raw)
    for i,fcsv in enumerate(pr_fileList):
        df = pd.read_csv(sav_psg_raw+fcsv)

        time_list = []
        fname = fcsv.split('.')[0]
        time_head = df.columns[0]
        time_head_rr = df.columns[1]
        df_length = int(df[time_head].shape[0] / 3)
        test_time = fcsv.split('_')[1][:4] + '-' + fcsv.split('_')[1][4:6] + '-' + fcsv.split('_')[1][6:]
        startTime = test_time + ' ' + time_head
        time_list.append(startTime)


        for i in range(df_length):
            endTime = (datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
            startTime = endTime# 参数days=1（天+1） 可以换成 minutes=1（分钟+1）、seconds=1（秒+1）
            time_list.append(endTime)

        pr_list = df[time_head].tolist()[::3];num_pop(time_list,pr_list)

        rslt = {'time': time_list, 'hr': pr_list};dframe = pd.DataFrame(data=rslt)

        dframe.to_csv((sav_psg_pr + fname+ '.csv'),index=False,header=['time','heart_rate'])

    # slp批量仿真数据，保存心率CSV文件
    slp_hr_br_transfrom(cat_slp,sav_slp_hr,0)

    # 心率准确性计算脚本
    psg_slp_heart_cal(sav_slp_hr, sav_psg_pr, src_txt, src_img)


else:


    psg_rr_transfrom(cat_psg, sav_psg_rr)
    psg_rr_nameList = os.listdir(sav_psg_rr)
    psg_pr_nameList = os.listdir(sav_psg_raw)
    psg_pr_rr_nameList = []
    psg_idList = []
    for i,prr_name in enumerate(psg_rr_nameList):

        filename = 'PSG'
        # pr_fileName = psg_pr_nameList[i]
        # csv_path = sav_psg_raw + pr_fileName.replace('脉率','')


        csv_path = sav_psg_raw + prr_name
        prr_path = sav_psg_rr + prr_name
        save_path = sav_psg_pr_rr
        psg_id = prr_name.split('.')[0].split('_')[0];psg_idList.append(psg_id)
        test_time = prr_name.split('_')[1][:4]+'-'+prr_name.split('_')[1][4:6]+'-'+prr_name.split('_')[1][6:]
        save_time = prr_name.split('_')[1][:4]+'_'+prr_name.split('_')[1][4:6]+'_'+prr_name.split('_')[1][6:]




        df = pd.read_csv(csv_path)
        time_head = df.columns[0]
        df_length = int(df[time_head].shape[0]/3)

        rr = pd.read_csv(prr_path)
        time_head_rr = rr.columns[1]

        time_list = []
        startTime = test_time + ' ' + time_head
        time_list.append(startTime)
        for i in range(df_length):
            endTime = (datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
            startTime = endTime# 参数days=1（天+1） 可以换成 minutes=1（分钟+1）、seconds=1（秒+1）
            time_list.append(endTime)

        pr_list = df[time_head].tolist()[::3]
        rr_list = rr[time_head_rr].tolist()


        pr_len = []
        rr_len = []
        time_len = []
        itLen = len_compare(pr_list,rr_list)
        for r in range(itLen):
            time_len.append(time_list[r])
            pr_len.append(pr_list[r])
            rr_len.append(rr_list[r])


        rslt = {'time':time_len, 'hr':pr_len, 'br':rr_len}
        dframe = pd.DataFrame(data=rslt)


        savetime = '_'+save_time+'_' + str(time_head).replace(':','')[:2]+\
                   '_'+str(time_head).replace(':','')[2:4]+\
                   '_'+str(time_head).replace(':','')[4:]+'.csv'
        suffix_file = psg_id + savetime
        savefile = save_path + suffix_file
        psg_pr_rr_nameList.append(suffix_file)
        dframe.to_csv((savefile), index=0)

    # slp批量仿真数据,保存心率、呼吸率CSV文件
    slp_hr_br_transfrom(cat_slp, src_slp, 2)

    # 心率、呼吸率准确性计算
    psg_slp_heart_breath_cal(src_slp,src_psg,src_txt,src_img)

