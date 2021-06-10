#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
# @author : biao chen
# @Email : chenbiao@sleepace.net
# @Project : Python_Files
# @File : utils.py
# @Software: PyCharm
# @Time : 2021/5/20 下午7:42
"""
import os
import sys
import time
import struct
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback
from pathlib import Path
from itertools import chain
from datetime import datetime

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

pd.set_option("display.max_columns", None)
# 相应的我们可以设置显示的最大行数
pd.set_option("display.max_rows", None)

# function: byte2int
def byte2int(data, mode="u16"):
    dbyte = bytearray(data)
    darray = []

    i = 0
    while i < len(dbyte):

        if "u8" == mode:
            darray.append(dbyte[i])
            i = i + 1
        elif "u16" == mode:
            darray.append(dbyte[i] | dbyte[i + 1] << 8)
            i = i + 2

    return darray


# end: byte2int

# function: byte2float
def byte2float(data, mode="float"):
    darray = []

    i = 0
    if "float" == mode:
        while i < len(data):
            fx = struct.unpack("f", data[i : i + 4])
            darray.append(fx)
            i = i + 4
    elif "double" == mode:
        while i < len(data):
            dx = struct.unpack("d", data[i : i + 8])
            darray.append(dx)
            i = i + 8

    return darray


# end: byte2float


def read_bytefile(path, folder, file, mode="u8"):
    fname = path + folder + file
    f = open(fname, "rb")
    dtmp = f.read()
    global rslt
    if "u8" == mode:
        rslt = byte2int(dtmp, mode="u8")
    if "u16" == mode:
        rslt = byte2int(dtmp, mode="u16")
    if "float" == mode:
        rslt = byte2float(dtmp, mode="float")
    if "double" == mode:
        rslt = byte2float(dtmp, mode="double")
    return rslt


# 向sheet中写入一行数据
def insertOne(value, sheet):
    sheet.append(value)


def read_raw(src_dir, fname):
    bcg, gain = [], []
    fname = src_dir + fname
    f = open(fname, "rb")
    dtmp = f.read()
    dbyte = bytearray(dtmp)

    i = 0
    while i < len(dbyte):
        bcg.append(dbyte[i] | dbyte[i + 1] << 8)
        gain.append(dbyte[i + 2])
        i = i + 3
    return bcg, gain


def read_wgt(src_dir, fname):
    wgt = []
    fname = src_dir + fname
    f = open(fname, "rb")
    dtmp = f.read()
    dbyte = bytearray(dtmp)

    i = 0
    while i < len(dbyte):
        wgt.append(dbyte[i + 1] | dbyte[i] << 8)
        i = i + 2
    return wgt


def time2stamp(cmnttime):  # 转时间戳函数
    # 转为时间数组
    timeArray = time.strptime(cmnttime, "%Y-%m-%d %H:%M:%S")
    # 转为时间戳
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


def stamp2time(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def day2stamp(cmnttime):  # 转时间戳函数
    # 转为时间数组
    timeArray = time.strptime(cmnttime, "%Y-%m-%d")
    # 转为时间戳
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


def stamp2day(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d", timeArray)
    return otherStyleTime


def hour2stamp(cmnttime):  # 转时间戳函数
    # 转为时间数组
    timeArray = time.strptime(cmnttime, "%Y-%m-%d %H:%M")
    # 转为时间戳
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


def stamp2hour(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M", timeArray)
    return otherStyleTime


def time2datetime(tranTime, pList, flag):
    if flag == 1:
        tdelta, sdelta, startstamp = 60, 1, int(time2stamp(tranTime))
        t = [datetime.fromtimestamp(startstamp + t * tdelta) for t in range(len(pList))]
        return t

    elif flag == 0:
        # startstamp = int(time2stamp(tranTime))
        famTime = [datetime.fromisoformat(t) for t in pList]
        return famTime


def quest_time_extract(num_spl, quest_outbed, slp_awTim):

    num_slp0 = num_spl[0]
    num_slp2 = num_spl[:2]
    aslp_day = stamp2day(day2stamp(slp_awTim) - 86400)
    awak_day = slp_awTim

    if len(num_spl) == 6:
        outbed_stamp = "0" + num_spl[0] + ":" + num_spl[1:3] + ":00"
        if int(num_slp0) >= 19 and int(num_slp0) <= 23:
            outbed_stamp = aslp_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)
        elif int(num_slp0) >= 0 and int(num_slp0) <= 8:
            outbed_stamp = awak_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)

    elif len(num_spl) == 4:
        outbed_stamp = num_spl[:2] + ":" + num_spl[2:] + ":00"
        if int(num_slp2) >= 19 and int(num_slp2) <= 23:
            outbed_stamp = aslp_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)

        elif int(num_slp2) >= 0 and int(num_slp2) <= 8:
            outbed_stamp = awak_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)

    elif len(num_spl) == 3:
        outbed_stamp = "0" + num_spl[0] + ":" + num_spl[1:] + ":00"
        if int(num_slp0) >= 19 and int(num_slp0) <= 23:
            outbed_stamp = aslp_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)

        elif int(num_slp0) >= 0 and int(num_slp0) <= 8:
            outbed_stamp = awak_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)

    elif len(num_spl) == 2:
        outbed_stamp = "0" + num_spl[0] + ":" + "00" + ":00"
        if int(num_slp0) >= 19 and int(num_slp0) <= 23:
            outbed_stamp = aslp_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)

        elif int(num_slp0) >= 0 and int(num_slp0) <= 8:
            outbed_stamp = awak_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)

    elif len(num_spl) == 1:
        outbed_stamp = "0" + num_spl + ":" + "00" + ":00"
        if int(num_spl) >= 19 and int(num_spl) <= 23:
            outbed_stamp = aslp_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)

        elif int(num_spl) >= 0 and int(num_spl) <= 8:
            outbed_stamp = awak_day + " " + outbed_stamp
            quest_outbed.append(outbed_stamp)


def diff_acl(slpList, psgList):
    fslp_diff = int(abs(time2stamp(str(psgList)) - time2stamp(str(slpList))) / 60)
    return fslp_diff


def num_pop(num1: list, num2: list):
    if len(num1) > len(num2):
        lenDiff = len(num1) - len(num2)
        for i in range(lenDiff):
            num1.pop()

    elif len(num2) > len(num1):
        lenDiff = len(num2) - len(num1)
        for i in range(lenDiff):
            num2.pop()


def num3_pop(num1: list, num2: list, num3: list):
    num2 = [str(i) for i in range(len(num2))]
    num3 = [str(i) for i in range(len(num3))]

    maxLen = max(len(num1), len(num2), len(num3))
    minLen = min(len(num1), len(num2), len(num3))
    plen = maxLen - minLen
    new_num1, new_num2, new_num3 = 0, 0, 0
    for i in range(maxLen):

        if len(num1) == maxLen:
            new_num1 = num1[:-plen]

        elif len(num2) == maxLen:
            new_num2 = num2[:-plen]

        elif len(num3) == maxLen:
            new_num3 = num3[:-plen]

    return new_num1, new_num2, new_num3


def len_compare(pr_list: list, rr_list: list):
    if len(pr_list) > len(rr_list):
        return len(rr_list)
    elif len(pr_list) < len(rr_list):
        return len(pr_list)


def path_concat(sub_dir, pathName):
    _path = str(sub_dir.joinpath(pathName)) + "/"
    return _path


def is_empty_file_3(file_path: str):
    assert isinstance(file_path, str), f"file_path参数类型不是字符串类型: {type(file_path)}"
    p = Path(file_path)
    assert p.is_file(), f"file_path不是一个文件: {file_path}"

    return p.stat().st_size == 0


def dir_empty(dir_path):
    try:
        next(os.scandir(dir_path))
        return False
    except StopIteration:
        return True


def select_num(df1, df2):
    # num_requried = 0
    hr_lower_limit = df1["hr"].map(lambda x: x != 0)
    hr_upper_limit = df1["hr"].map(lambda x: x != 255)
    br_lower_limit = df1["br"].map(lambda x: x != 0)
    br_upper_limit = df1["br"].map(lambda x: x != 255)
    pr_lower_limit = df2["pr"].map(lambda x: x != 0)
    pr_upper_limit = df2["pr"].map(lambda x: x != 255)
    rr_lower_limit = df2["rr"].map(lambda x: x != 0)
    rr_upper_limit = df2["rr"].map(lambda x: x != 255)

    df1 = df1[
        (hr_lower_limit & hr_upper_limit & br_lower_limit & br_upper_limit)
        & (pr_lower_limit & pr_upper_limit & rr_lower_limit & rr_upper_limit)
    ]

    df2 = df2[
        (hr_lower_limit & hr_upper_limit & br_lower_limit & br_upper_limit)
        & (pr_lower_limit & pr_upper_limit & rr_lower_limit & rr_upper_limit)
    ]

    df1 = df1.reset_index(drop=True)  # 重新给索引
    df2 = df2.reset_index(drop=True)  # 重新给索引
    return df1, df2


def minute_mean(df, cname, stime):
    # 计算每分钟SLP的心率、呼吸率
    hr_min_list = []
    slp_time_min_list = []
    df_min = int(len(df[cname]) / 60)  # 数据共多少分钟

    for i in range(df_min):
        hr_min_len = (i + 1) * 60
        num = 0
        temp = 0
        slp_time_min = stime + hr_min_len
        for j in df[cname][hr_min_len - 60 : hr_min_len]:
            if j != 0 and j != 255:
                num += 1
                temp += j
        if num > 0:
            res = int(temp / num)
            hr_min_list.append(res)
        if num == 0:
            hr_min_list.append(0)
        slp_time_min_list.append(slp_time_min)

    # rslt = {'time':slp_time_min_list,'hr':hr_min_list,'br':br_min_list}
    # df_clean = pd.DataFrame(data=rslt)
    return slp_time_min_list, hr_min_list


def file_exist(my_file):
    txt_list = []
    if Path(my_file).is_file() == False:
        Path(my_file).touch()
        return txt_list


def Heart_rate_accuracy_calculat(PR, HR, src_txt, fcsv):
    PR = PR[PR.map(lambda x: x > 0)]
    HR = HR[HR.map(lambda x: x > 0)]
    PR = PR.reset_index(drop=True)  # 重新给索引
    HR = HR.reset_index(drop=True)  # 重新给索引

    diff_hr = PR - HR
    diff_hr_cnt = 0
    try:
        diff_hr_pre = abs(diff_hr) / PR
        diff_hr_pre = diff_hr_pre.dropna()
        diff_hr_pre = diff_hr_pre * 100
        for i, val in enumerate(diff_hr):
            if i <= len(PR):
                if abs(val) <= PR[i] * 0.1 or abs(val) <= 5:
                    diff_hr_cnt += 1

        hr_mean = round(np.mean(abs(diff_hr)), 2)
        hr_std = round(np.std(abs(diff_hr), ddof=1), 2)
        if len(diff_hr_pre) == 0:
            print(traceback.print_exc())
        else:
            acc_hr = diff_hr_cnt / len(diff_hr_pre)
            txt_content = (
                fcsv
                + " 心率准确性[%d / %d]: %.2f %%"
                % (
                    diff_hr_cnt,
                    len(diff_hr_pre),
                    round(acc_hr * 100, 2),
                )
                + " 心率误差：",
                str(hr_mean) + "±" + str(hr_std),
            )
            f = open(src_txt + "accuracy.txt", "a")
            f.write((str(txt_content) + "\r"))

            return acc_hr
    except:
        print(traceback.print_exc())


def Respiration_rate_accuracy_calculat(RR, br, src_txt, fcsv):
    RR = RR[RR.map(lambda x: x > 0)]
    br = br[br.map(lambda x: x > 0)]
    RR = RR.reset_index(drop=True)  # 重新给索引
    br = br.reset_index(drop=True)  # 重新给索引

    try:
        # 计算呼吸率准确性
        diff_br_pre = abs(RR - br)
        diff_br_pre = diff_br_pre.dropna()
        diff_br_cnt = 0
        for i in diff_br_pre:
            if i <= 2:
                diff_br_cnt += 1

        br_mean = round(np.mean(abs(diff_br_pre)), 2)
        br_std = round(np.std(abs(diff_br_pre), ddof=1), 2)
        if len(diff_br_pre) == 0:
            print(traceback.print_exc())
        else:
            acc_br = diff_br_cnt / len(diff_br_pre)
            txt_content = (
                fcsv
                + " 呼吸率准确性[%d / %d]: %.2f %%"
                % (
                    diff_br_cnt,
                    len(diff_br_pre),
                    round(acc_br * 100, 2),
                )
                + " 呼吸率误差：",
                str(br_mean) + "±" + str(br_std),
            )
            f = open(src_txt + "accuracy.txt", "a")
            f.write((str(txt_content) + "\r"))

            return acc_br
    except:
        print(traceback.print_exc())


def draw_PR_save(PR, slp_hr, time_offset, img_dir, fcsv, acc_flag):
    # 作图
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    # 配置横坐标日期显示#格式#间隔
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y%m/%d %H:%M:%S"))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))

    if len(PR) > len(time_offset):
        PR = PR[:-1]

    ax1 = plt.subplot(412)
    plt.plot(time_offset, PR, "r-", label="PSG")
    plt.plot(time_offset, slp_hr, "b-", label="智能枕头")
    plt.title("心率对比(bpm)", fontsize=9)
    plt.legend(loc="upper right")
    plt.setp(ax1.get_xticklabels(), visible=False, fontsize=9)
    # plt.xlim(time_offset[0], time_offset[-1])
    plt.ylim(40, 100)

    f = plt.gcf()  # 获取当前图像
    if acc_flag == 1:
        f.savefig(img_dir + "err_img/" + fcsv + ".png", bbox_inches="tight")
    elif acc_flag == 0:
        f.savefig(img_dir + "nor_img/" + fcsv + ".png", bbox_inches="tight")
    f.clear()  # 释放内存


def draw_PR_RR_save(PR, RR, slp_hr, slp_br, time_offset, img_dir, fcsv, acc_flag):
    # 作图
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    # fig.suptitle(fname)
    # 配置横坐标日期显示#格式#间隔
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y%m/%d %H:%M:%S"))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))

    if len(PR) > len(time_offset):
        PR = PR[:-1]

    if len(RR) > len(time_offset):
        RR = RR[:-1]

    print(len(time_offset), len(PR))
    print(time_offset)

    ax1 = plt.subplot(412)
    plt.plot(time_offset, PR, "r-", label="PSG")
    plt.plot(time_offset, slp_hr, "b-", label="智能枕头")
    plt.title("心率对比(bpm)", fontsize=9)
    plt.legend(loc="upper right")
    plt.setp(ax1.get_xticklabels(), visible=False, fontsize=9)
    # plt.xlim(time_offset[0], time_offset[-1])
    plt.ylim(40, 100)

    ax2 = plt.subplot(413, sharex=ax1)
    plt.plot(time_offset, RR, "r-", label="PSG")
    plt.plot(time_offset, slp_br, "b-", label="智能枕头")
    plt.title("呼吸率对比(rpm)", fontsize=9)
    plt.legend(loc="upper right")
    plt.setp(ax2.get_xticklabels(), visible=True, fontsize=9)
    plt.xticks()
    # plt.xlim(time_offset[0], time_offset[-1])
    plt.ylim(5, 35)

    f = plt.gcf()  # 获取当前图像
    if acc_flag == 1:
        f.savefig(img_dir + "err_img/" + fcsv + ".png", bbox_inches="tight")
    elif acc_flag == 0:
        f.savefig(img_dir + "nor_img/" + fcsv + ".png", bbox_inches="tight")
    # f.figlegend()
    f.clear()  # 释放内存


def slp_hr_br_transfrom(cat_dir, save_dir, flag):

    # slp批量仿真数据转成csv文件

    flist = os.listdir(cat_dir + "hr_sec/")
    for fcsv in flist[:]:
        fname = fcsv.split(".")[0]
        hr_list = read_bytefile(cat_dir, "hr_sec/", fcsv, mode="u8")
        br_list = read_bytefile(cat_dir, "br_sec/", fcsv, mode="u8")

        tdelta, startstamp = 1 / 100.0, int(fcsv.split("_")[-1].split(".")[0])
        time_list = [startstamp + t for t in range(len(hr_list))]

        if flag == 0:
            rslt = {"time": time_list, "heart_rate": hr_list}
            df = pd.DataFrame(data=rslt)
            df.to_csv(
                (save_dir + fname + ".csv"), index=False, header=["time", "heart_rate"]
            )
        elif flag == 1:
            rslt = {"time": time_list, "breath_rate": br_list}

            df = pd.DataFrame(data=rslt)
            df.to_csv(
                (save_dir + fname + ".csv"), index=False, header=["time", "breath_rate"]
            )
        elif flag == 2:
            rslt = {"time": time_list, "heart_rate": hr_list, "breath_rate": br_list}
            df = pd.DataFrame(data=rslt)
            df.to_csv(
                (save_dir + fname + ".csv"),
                index=False,
                header=["time", "heart_rate", "breath_rate"],
            )


def psg_slp_heart_cal(src_slp, src_psg, src_txt, src_img):
    ############心率准确性脚本计算############
    slp_flist = os.listdir(src_slp)
    psg_flist = os.listdir(src_psg)
    txt_list = []
    my_file = src_txt + "setime.txt"
    acc_file = src_txt + "accuracy.txt"

    for i, fcsv in enumerate(slp_flist):

        simg_name = fcsv.split(".")[0]

        data_slp = pd.read_csv(src_slp + fcsv)
        print(fcsv, psg_flist[i])
        data_psg = pd.read_csv(src_psg + psg_flist[i])
        data_slp.columns = ["time", "hr"]
        data_psg.columns = ["time", "pr"]
        time_set = [
            data_slp["time"].tolist()[0],
            time2stamp(data_psg["time"].tolist()[0]),
            data_slp["time"].tolist()[-1],
            time2stamp(data_psg["time"].tolist()[-1]),
        ]

        start_time = time_set[0] - time_set[1]
        end_time = time_set[2] - time_set[3]

        slp_time_len = data_slp["time"].tolist()[-1] - data_slp["time"].tolist()[0]
        psg_time_len = time2stamp(data_psg["time"].tolist()[-1]) - time2stamp(
            data_psg["time"].tolist()[0]
        )

        if start_time < 0:
            file_start = time_set[1]
        else:
            file_start = time_set[0]

        if end_time < 0:
            file_end = time_set[2]
        else:
            file_end = time_set[3]

        data_psg["timestamp"] = data_psg["time"].apply(lambda x: time2stamp(x))

        print(
            "开始区间：", file_start, "结束区间：", file_end, "公共区间长度：", (file_end - file_start)
        )

        slp_sind = data_slp[data_slp["time"] == file_start].index.tolist()[0]
        slp_eind = data_slp[data_slp["time"] == file_end].index.tolist()[0]
        slp_clist = data_slp[slp_sind : slp_eind + 1]

        psg_sind = data_psg[data_psg["timestamp"] == file_start].index.tolist()[0]
        psg_eind = data_psg[data_psg["timestamp"] == file_end].index.tolist()[0]
        psg_clist = data_psg[psg_sind : psg_eind + 1]

        hr_time, hr_list = minute_mean(slp_clist, "hr", file_start)
        pr_time, pr_list = minute_mean(psg_clist, "pr", file_start)

        rslt_slp = {"time": hr_time, "hr": hr_list}
        clean_slp = pd.DataFrame(data=rslt_slp)

        rslt_psg = {"time": pr_time, "pr": pr_list}
        clean_psg = pd.DataFrame(data=rslt_psg)

        time = clean_slp["time"]
        HR = clean_slp["hr"]
        PR = clean_psg["pr"]

        acc_hr = Heart_rate_accuracy_calculat(PR, HR, src_txt, fcsv)

        time_offset = [datetime.fromtimestamp(i) for i in time]
        # 准备原始SLP心率、呼吸数据
        slp_hr = pd.Series(list(HR), index=time_offset)

        if len(time_offset) > 0:
            acc_flag = 0
            if acc_hr < 0.9:
                acc_flag = 1
                draw_PR_save(PR, slp_hr, time_offset, src_img, simg_name, acc_flag)
            else:
                draw_PR_save(PR, slp_hr, time_offset, src_img, simg_name, acc_flag)

            if Path(my_file).is_file() == False:
                Path(my_file).touch()

            if Path(my_file).exists():
                size = os.path.getsize(my_file)
                if size > 100:
                    os.remove(my_file)
                    Path(my_file).touch()
                elif size == 0:
                    time_diff = file_end - file_start
                    txt_content = (
                        fcsv
                        + " 起始时间："
                        + str(file_start)
                        + " 结束时间："
                        + str(file_end)
                        + " 时间长度："
                        + str(time_diff)
                    )
                    txt_list.append(txt_content)

    for i, val in enumerate(txt_list):
        f = open(my_file, "a")
        f.write((str(val) + "\r"))
        f.close()


def psg_slp_heart_breath_cal(src_slp, src_psg, src_txt, src_img, flag):
    ############心率、呼吸率准确性计算脚本############

    if flag == 0:
        slp_flist = os.listdir(src_slp)
        psg_flist = os.listdir(src_psg)

        slp_idList = [i.split(".")[0].split("_")[0] for i in slp_flist]
        psg_idList = [i.split(".")[0].split("_")[0] for i in psg_flist]
        txt_list = []
        my_file = src_txt + "setime.txt"
        acc_file = src_txt + "accuracy.txt"

        for i, fcsv in enumerate(slp_flist):
            # print(slp_idList[i],psg_idList[i])
            j = psg_idList.index(slp_idList[i])
            simg_name = fcsv.split(".")[0]
            data_slp = pd.read_csv(src_slp + fcsv)
            data_psg = pd.read_csv(src_psg + psg_flist[j])
            data_slp.columns = ["time", "hr", "br"]
            data_psg.columns = ["time", "pr", "rr"]

            time_set = [
                data_slp["time"].tolist()[0],
                time2stamp(data_psg["time"].tolist()[0]),
                data_slp["time"].tolist()[-1],
                time2stamp(data_psg["time"].tolist()[-1]),
            ]

            start_time = time_set[0] - time_set[1]
            end_time = time_set[2] - time_set[3]

            slp_time_len = data_slp["time"].tolist()[-1] - data_slp["time"].tolist()[0]
            psg_time_len = time2stamp(data_psg["time"].tolist()[-1]) - time2stamp(
                data_psg["time"].tolist()[0]
            )

            if start_time < 0:
                file_start = time_set[1]
            else:
                file_start = time_set[0]

            if end_time < 0:
                file_end = time_set[2]
            else:
                file_end = time_set[3]

            data_psg["timestamp"] = data_psg["time"].apply(lambda x: time2stamp(x))

            print(
                "开始区间：",
                file_start,
                "结束区间：",
                file_end,
                "公共区间长度：",
                (file_end - file_start),
            )

            slp_sind = data_slp[data_slp["time"] == file_start].index.tolist()[0]
            slp_eind = data_slp[data_slp["time"] == file_end].index.tolist()[0]
            slp_clist = data_slp[slp_sind : slp_eind + 1]

            psg_sind = data_psg[data_psg["timestamp"] == file_start].index.tolist()[0]
            psg_eind = data_psg[data_psg["timestamp"] == file_end].index.tolist()[0]
            psg_clist = data_psg[psg_sind : psg_eind + 1]

            hr_time, hr_list = minute_mean(slp_clist, "hr", file_start)
            br_time, br_list = minute_mean(slp_clist, "br", file_start)
            pr_time, pr_list = minute_mean(psg_clist, "pr", file_start)
            rr_time, rr_list = minute_mean(psg_clist, "rr", file_start)

            rslt_slp = {"time": hr_time, "hr": hr_list, "br": br_list}
            clean_slp = pd.DataFrame(data=rslt_slp)

            rslt_psg = {"time": pr_time, "pr": pr_list, "rr": rr_list}
            clean_psg = pd.DataFrame(data=rslt_psg)

            time = clean_slp["time"]
            HR = clean_slp["hr"]
            PR = clean_psg["pr"]
            BR = clean_slp["br"]
            RR = clean_psg["rr"]

            acc_hr = Heart_rate_accuracy_calculat(PR, HR, src_txt, fcsv)
            acc_br = Respiration_rate_accuracy_calculat(RR, BR, src_txt, fcsv)

            time_offset = [datetime.fromtimestamp(i) for i in time]
            # 准备原始SLP心率、呼吸数据
            slp_hr = pd.Series(list(HR), index=time_offset)
            slp_br = pd.Series(list(BR), index=time_offset)

            if len(time_offset) > 0:
                acc_flag = 0
                if acc_hr != None and acc_br != None:

                    if acc_hr < 0.9 or acc_br < 0.9:
                        acc_flag = 1
                        draw_PR_RR_save(
                            PR,
                            RR,
                            slp_hr,
                            slp_br,
                            time_offset,
                            src_img,
                            simg_name,
                            acc_flag,
                        )
                    else:
                        draw_PR_RR_save(
                            PR,
                            RR,
                            slp_hr,
                            slp_br,
                            time_offset,
                            src_img,
                            simg_name,
                            acc_flag,
                        )

                    if Path(my_file).is_file() == False:
                        Path(my_file).touch()

                    if Path(my_file).exists():
                        size = os.path.getsize(my_file)
                        if size > 100:
                            os.remove(my_file)
                            Path(my_file).touch()
                        elif size == 0:
                            time_diff = file_end - file_start
                            txt_content = (
                                fcsv
                                + " 起始时间："
                                + str(file_start)
                                + " 结束时间："
                                + str(file_end)
                                + " 时间长度："
                                + str(time_diff)
                            )
                            txt_list.append(txt_content)

        for i, val in enumerate(txt_list):
            f = open(my_file, "a")
            f.write((str(val) + "\r"))
            f.close()

    elif flag == 1:
        slp_flist = os.listdir(src_slp)
        psg_flist = os.listdir(src_psg)

        slp_idList = [i.split(".")[0].split("_")[0] for i in slp_flist]
        psg_idList = [i.split(".")[0].split("_")[0].lstrip("0") for i in psg_flist]
        txt_list = []
        my_file = src_txt + "setime.txt"
        acc_file = src_txt + "accuracy.txt"

        for i, fcsv in enumerate(slp_flist):
            j = psg_idList.index(slp_idList[i])
            simg_name = fcsv.split(".")[0]
            data_slp = pd.read_csv(src_slp + fcsv)
            data_psg = pd.read_csv(src_psg + psg_flist[j])
            data_slp.columns = ["time", "hr", "br"]
            data_psg.columns = ["time", "pr", "rr"]

            time_set = [
                data_slp["time"].tolist()[0],
                hour2stamp(data_psg["time"].tolist()[0]),
                data_slp["time"].tolist()[-1],
                hour2stamp(data_psg["time"].tolist()[-1]),
            ]

            start_time = time_set[0] - time_set[1]
            end_time = time_set[2] - time_set[3]
            # slp_start_len,psg_start_len = len(data_slp['time'].tolist()[0]),len(time2stamp(data_psg['time'].tolist()[0]))
            # slp_end_len,psg_end_len = len(data_slp['time'].tolist()[-1]),len(time2stamp(data_psg['time'].tolist()[-1]))

            slp_time_len = data_slp["time"].tolist()[-1] - data_slp["time"].tolist()[0]
            psg_time_len = hour2stamp(data_psg["time"].tolist()[-1]) - hour2stamp(
                data_psg["time"].tolist()[0]
            )

            # print(slp_time_len, psg_time_len)
            # print(data_slp['time'].tolist()[0], time2stamp(data_psg['time'].tolist()[0]))
            # print(data_slp['time'].tolist()[-1], time2stamp(data_psg['time'].tolist()[-1]))

            if start_time < 0:
                file_start = time_set[1]
            else:
                file_start = time_set[0]

            if end_time < 0:
                file_end = time_set[2]
            else:
                file_end = time_set[3]

            print(time_set[1], time_set[0])

            data_psg["timestamp"] = data_psg["time"].apply(lambda x: hour2stamp(x))

            print(
                "开始区间：",
                file_start,
                "结束区间：",
                file_end,
                "公共区间长度：",
                (file_end - file_start),
            )

            slp_sind = data_slp[data_slp["time"] == file_start].index.tolist()[0]
            slp_eind = data_slp[data_slp["time"] == file_end].index.tolist()[0]
            slp_clist = data_slp[slp_sind : slp_eind + 1]

            psg_sind = data_psg[data_psg["timestamp"] == file_start].index.tolist()[0]
            psg_eind = data_psg[data_psg["timestamp"] == file_end].index.tolist()[0]
            psg_clist = data_psg[psg_sind : psg_eind + 1]

            hr_time, hr_list = minute_mean(slp_clist, "hr", file_start)
            br_time, br_list = minute_mean(slp_clist, "br", file_start)
            pr_time, pr_list = minute_mean(psg_clist, "pr", file_start)
            rr_time, rr_list = minute_mean(psg_clist, "rr", file_start)

            rslt_slp = {"time": hr_time, "hr": hr_list, "br": br_list}
            clean_slp = pd.DataFrame(data=rslt_slp)

            rslt_psg = {"time": pr_time, "pr": pr_list, "rr": rr_list}
            clean_psg = pd.DataFrame(data=rslt_psg)

            time = clean_slp["time"]
            HR = clean_slp["hr"]
            PR = clean_psg["pr"]
            BR = clean_slp["br"]
            RR = clean_psg["rr"]

            acc_hr = Heart_rate_accuracy_calculat(PR, HR, src_txt, fcsv)
            acc_br = Respiration_rate_accuracy_calculat(RR, BR, src_txt, fcsv)

            time_offset = [datetime.fromtimestamp(i) for i in time]
            # 准备原始SLP心率、呼吸数据
            slp_hr = pd.Series(list(HR), index=time_offset)
            slp_br = pd.Series(list(BR), index=time_offset)

            if len(time_offset) > 0:
                acc_flag = 0
                if acc_hr < 0.9 or acc_br < 0.9:
                    acc_flag = 1
                    draw_PR_RR_save(
                        PR,
                        RR,
                        slp_hr,
                        slp_br,
                        time_offset,
                        src_img,
                        simg_name,
                        acc_flag,
                    )
                else:
                    draw_PR_RR_save(
                        PR,
                        RR,
                        slp_hr,
                        slp_br,
                        time_offset,
                        src_img,
                        simg_name,
                        acc_flag,
                    )

                if Path(my_file).is_file() == False:
                    Path(my_file).touch()

                if Path(my_file).exists():
                    size = os.path.getsize(my_file)
                    if size > 100:
                        os.remove(my_file)
                        Path(my_file).touch()
                    elif size == 0:
                        time_diff = file_end - file_start
                        txt_content = (
                            fcsv
                            + " 起始时间："
                            + str(file_start)
                            + " 结束时间："
                            + str(file_end)
                            + " 时间长度："
                            + str(time_diff)
                        )
                        txt_list.append(txt_content)

        for i, val in enumerate(txt_list):
            f = open(my_file, "a")
            f.write((str(val) + "\r"))
            f.close()


def psg_rr_transfrom(cat_dir, save_dir):

    # psg批量仿真数据转成csv文件
    flist = os.listdir(cat_dir + "br_sec/")
    for fcsv in flist[:]:
        fname = fcsv.split(".")[0]
        br_list = read_bytefile(cat_dir, "br_sec/", fcsv, mode="u8")

        tdelta, startstamp = 1 / 100.0, int(fcsv.split("_")[-1].split(".")[0])
        time_list = [startstamp + t for t in range(len(br_list))]
        rslt = {"time": time_list, "breath_rate": br_list}

        df = pd.DataFrame(data=rslt)
        df.to_csv(
            (save_dir + fname + ".csv"), index=False, header=["time", "breath_rate"]
        )


def read_summary(path, folder, file):
    fname = path + folder + file
    f = open(fname, "rb")
    dtmp = f.read()
    dtmp = bytearray(dtmp)

    mean_hrate = dtmp[0] | dtmp[1] << 8  # 平均心率
    mean_brate = dtmp[2] | dtmp[3] << 8  # 平均呼吸率
    fallasleeptime = dtmp[4] | dtmp[5] << 8  # 入睡时刻
    wakeuptime = dtmp[6] | dtmp[7] << 8  # 清醒时刻
    offbed_cnt = dtmp[8] | dtmp[9] << 8  # 离床次数
    turnover_cnt = dtmp[10] | dtmp[11] << 8  # 翻身次数
    bodymove_cnt = dtmp[12] | dtmp[13] << 8  # 体动次数
    heartstop_cnt = dtmp[14] | dtmp[15] << 8  # 心跳暂停次数
    respstop_cnt = dtmp[16] | dtmp[17] << 8  # 呼吸暂停次数
    deepsleep_per = dtmp[18] | dtmp[19] << 8  # 深睡比例
    remsleep_per = dtmp[20] | dtmp[21] << 8  # 中睡比例
    lightsleep_per = dtmp[22] | dtmp[23] << 8  # 浅睡比例
    wakesleep_per = dtmp[24] | dtmp[25] << 8  # 清醒比例
    wakesleep_time = dtmp[26] | dtmp[27] << 8  # 清醒时长
    lightsleep_time = dtmp[28] | dtmp[29] << 8  # 浅睡时长
    remsleep_time = dtmp[30] | dtmp[31] << 8  # 中睡时长
    deepsleep_time = dtmp[32] | dtmp[33] << 8  # 深睡时长
    wake_off_cnt = dtmp[34] | dtmp[35] << 8  # 清醒（含离床）次数
    hrate_max = dtmp[36] | dtmp[37] << 8  # 最高心率
    brate_max = dtmp[38] | dtmp[39] << 8  # 最高呼吸率
    hrate_min = dtmp[40] | dtmp[41] << 8  # 最低心率
    brate_min = dtmp[42] | dtmp[43] << 8  # 最低呼吸率
    hrate_high_time = dtmp[44] | dtmp[55] << 8  # 心率过速时长
    hrate_low_time = dtmp[46] | dtmp[47] << 8  # 心率过缓时长
    brate_high_time = dtmp[48] | dtmp[49] << 8  # 呼吸过速时长
    brate_low_time = dtmp[50] | dtmp[51] << 8  # 呼吸过缓时长
    allsleep_time = dtmp[52] | dtmp[53] << 8  # 睡觉时长
    body_move = dtmp[54] | dtmp[55] << 8  # 躁动不安扣分
    off_bed = dtmp[56] | dtmp[57] << 8  # 离床扣分
    wake_cnt = dtmp[58] | dtmp[59] << 8  # 易醒扣分
    start_time = dtmp[60] | dtmp[61] << 8  # 睡太晚扣分
    fall_asleep = dtmp[62] | dtmp[63] << 8  # 难于入睡扣分
    perc_deep = dtmp[64] | dtmp[65] << 8  # 深睡不足扣分
    sleep_long = dtmp[66] | dtmp[67] << 8  # 睡时间过长扣分
    sleep_less = dtmp[68] | dtmp[69] << 8  # 睡眠时间过短扣分
    breath_stop = dtmp[70] | dtmp[71] << 8  # 呼吸暂停扣分
    heart_stop = dtmp[72] | dtmp[73] << 8  # 心跳暂停扣分
    hrate_low = dtmp[74] | dtmp[75] << 8  # 心跳过缓扣分
    hrate_high = dtmp[76] | dtmp[77] << 8  # 心跳过速扣分
    brate_low = dtmp[78] | dtmp[79] << 8  # 呼吸过缓扣分
    brate_high = dtmp[80] | dtmp[81] << 8  # 呼吸过速扣分
    benign_sleep = dtmp[82] | dtmp[83] << 8  # 良性睡眠分布扣分

    offset = dtmp[84] | dtmp[85] << 8
    data_len = dtmp[86] | dtmp[87] << 8
    start_stamp = dtmp[88] | dtmp[89] << 8 | dtmp[90] << 16 | dtmp[91] << 24

    print(start_stamp, start_stamp + fallasleeptime * 60)

    diff = (
        body_move
        + off_bed
        + wake_cnt
        + start_time
        + fall_asleep
        + perc_deep
        + sleep_long
        + sleep_less
        + breath_stop
        + heart_stop
        + hrate_low
        + hrate_high
        + brate_low
        + brate_high
        + benign_sleep
    )
    score = 100 - diff

    rslt = {"offset": offset, "len": data_len, "start_time": start_stamp}

    print("-----睡眠报告-----")
    print(">>> 睡眠比例")
    print(
        "睡眠时长：%d H %d min (入睡：%d, 清醒：%d)"
        % (allsleep_time / 60, allsleep_time % 60, fallasleeptime, wakeuptime)
    )
    print(
        "深睡时长：%d H %d min (%d%%) | 中睡时长：%d H %d min (%d%%) | 浅睡时长：%d H %d min (%d%%) | 清醒时长：%d H %d min (%d%%)"
        % (
            deepsleep_time / 60,
            deepsleep_time % 60,
            deepsleep_per,
            remsleep_time / 60,
            remsleep_time % 60,
            remsleep_per,
            lightsleep_time / 60,
            lightsleep_time % 60,
            lightsleep_per,
            wakesleep_time / 60,
            wakesleep_time % 60,
            wakesleep_per,
        )
    )
    print(">>> 呼吸心率")
    print("平均呼吸：%d bpm (min: %d, max: %d)" % (mean_brate, brate_min, brate_max))
    print("呼吸暂停：%d 次" % respstop_cnt)
    print(
        "呼吸过速：%d H %d min | 呼吸过缓：%d H %d min "
        % (
            brate_high_time / 60,
            brate_high_time % 60,
            brate_low_time / 60,
            brate_low_time % 60,
        )
    )
    print("平均心率：%d bpm (min: %d, max: %d)" % (mean_hrate, hrate_min, hrate_max))
    print(
        "心率过速：%d H %d min | 心率过缓：%d H %d min "
        % (
            hrate_high_time / 60,
            hrate_high_time % 60,
            hrate_low_time / 60,
            hrate_low_time % 60,
        )
    )
    print("心跳暂停：%d 次" % heartstop_cnt)
    print(">>> 体动翻身")
    print(
        "体动次数：%d | 翻身次数：%d | 离床次数：%d | 清醒次数：%d "
        % (bodymove_cnt, turnover_cnt, offbed_cnt, wake_off_cnt)
    )
    print(">>> 睡眠分数")
    print("整晚睡眠得分：", score)
    print("躁动不安扣分：", body_move)
    print("离床过多扣分：", off_bed)
    print("睡觉易醒扣分：", wake_cnt)
    print("睡觉太晚扣分：", start_time)
    print("难于入睡扣分：", fall_asleep)
    print("深睡不足扣分：", perc_deep)
    print("睡眠过长扣分：", sleep_long)
    print("睡眠过短扣分：", sleep_less)
    print("呼吸暂停扣分：", breath_stop)
    print("心跳暂停扣分：", heart_stop)
    print("心跳过缓扣分：", hrate_low)
    print("心跳过速扣分：", hrate_high)
    print("呼吸过缓扣分：", brate_low)
    print("呼吸过速扣分：", brate_high)
    print("良性睡眠扣分：", benign_sleep)
    print("----------------")

    return rslt
