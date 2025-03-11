import os
import re
import time
import shutil
import subprocess
import numpy as np
from bs4 import BeautifulSoup
# from model_summary import get_model_flops, get_model_activation

def basecall_calculate(opt, i):
    d = opt.test_option[i]
    cycle = d['cycle']
    with open(opt.settings) as fh:
        lines = fh.readlines()
    newlines = []
    for line in lines:
        newline = line
        if 'modparam.writeFqFilter' in line and line[0] != '#':
            newline = f'modparam.writeFqFilter = {opt.writeFqFilter}\n'
        if 'client.imgSizeX' in line and line[0] != '#':
            newline = f'client.imgSizeX = {d["size"][0]}\n'
        if 'client.imgSizeY' in line and line[0] != '#':
            newline = f'client.imgSizeY = {d["size"][1]}\n'
        if 'modparam.trackWidth' in line and line[0] != '#':
            if d['chip'] == 'DP88':
                newline = 'modparam.trackWidth = 1800\n'
            elif d['chip'] == 'DP84':
                newline = 'modparam.trackWidth = 1080\n'
        if 'trackPatternBlockSizeX' in line and line[0] != '#':
            if d['chip'] == 'DP88':
                newline = 'modparam._DP88.trackPatternBlockSizeX = 100 175 200 225 225 200 175 100 300\n'
            elif d['chip'] == 'DP84':
                newline = 'modparam._DP84.trackPatternBlockSizeX = 160 240 280 320 320 280 240 160 512\n'
        if 'trackPatternBlockSizeY' in line and line[0] != '#':
            if d['chip'] == 'DP88':
                newline = 'modparam._DP88.trackPatternBlockSizeY = 100 175 200 225 225 200 175 100 300\n'
            elif d['chip'] == 'DP84':
                newline = 'modparam._DP84.trackPatternBlockSizeY = 160 240 280 320 320 280 240 160 512\n'
        if 'chipPitch' in line and line[0] != '#':
            if d['chip'] == 'DP88':
                newline = 'modparam._DP88.chipPitch = 360\n'
            elif d['chip'] == 'DP84':
                newline = 'modparam._DP84.chipPitch = 360\n'
        if 'pixelRadius' in line and line[0] != '#':
            if d['chip'] == 'DP88':
                newline = 'modparam._DP88.pixelRadius = 1.0\n'
            elif d['chip'] == 'DP84':
                newline = 'modparam._DP84.pixelRadius = 1.1\n'
        if 'bleedingFactorR1' in line and line[0] != '#':
            if d['chip'] == 'DP88':
                newline = 'modparam._DP88.bleedingFactorR1 = 0.04 0.020 0.04 0.020 0.04 0.020 0.04 0.020\n'
            elif d['chip'] == 'DP84':
                newline = 'modparam._DP84.bleedingFactorR1 = 0.04 0.025 0.04 0.025 0.04 0.025 0.04 0.025\n'
        if 'bleedingFactorR2' in line and line[0] != '#':
            if d['chip'] == 'DP88':
                newline = 'modparam._DP88.bleedingFactorR2 = 0.08 0.030 0.08 0.030 0.08 0.030 0.08 0.030\n'
            elif d['chip'] == 'DP84':
                newline = 'modparam._DP84.bleedingFactorR2 = 0.05 0.025 0.05 0.025 0.05 0.025 0.05 0.025\n'
        newlines.append(newline)
    with open(opt.settings, 'w') as fh:
        fh.writelines(newlines)
    # process = subprocess.Popen(opt.processor, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # process = subprocess.Popen(opt.processor)
    time.sleep(5)
    if cycle == 51:
        readlength = ['SE50']
    elif cycle == 1:
        readlength = ['SE1']
    elif cycle == 8:
        readlength = ['SE8']
    elif cycle == 30:
        readlength = ['SE30']
    elif cycle == 10:
        readlength = ['SE10']
    elif cycle == 101:
        readlength = ['SE100']
    elif cycle == 202:
        readlength = ['PE100']
    elif cycle == 212:
        readlength = ['PE100_barcode']
    for k in readlength:
        report = fr'{d["chip"]}_{opt.model}_{i}_C{str(d["fov_C"]).zfill(3)}R{str(d["fov_R"]).zfill(3)}_{k}_filter_{opt.writeFqFilter}'
        report_path = fr'{opt.basecall_dir}/OutputFq/{report}/'
        if os.path.exists(report_path):
            # os.removedirs(report_path)
            shutil.rmtree(report_path, ignore_errors=True)
        if k == 'SE50':
            os.system(fr'{opt.client} {opt.output} 51 {d["fov_C"]} {d["fov_R"]} -S -N {report} -E 1')
        elif k == 'SE8':
            cmd_line = fr'{opt.client} {opt.output} 8 {d["fov_C"]} {d["fov_R"]} -S -N {report}'
            # print( cmd_line )
            opt.logger.info(fr'{cmd_line}')
            subprocess.call(cmd_line, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif k == 'SE30':
            cmd_line = fr'{opt.client} {opt.output} {cycle} {d["fov_C"]} {d["fov_R"]} -S -N {report}'
            opt.logger.info(fr'{cmd_line}')
            subprocess.call(cmd_line, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif k == 'SE1':
            cmd_line = fr'{opt.client} {opt.output} 1 {d["fov_C"]} {d["fov_R"]} -S -N {report}'
            opt.logger.info(fr'{cmd_line}')
            subprocess.call(cmd_line, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif k == 'SE10':
            cmd_line = fr'{opt.client} {opt.output} 10 {d["fov_C"]} {d["fov_R"]} -S -N {report}'
            # print( cmd_line )
            opt.logger.info(fr'{cmd_line}')
            # os.system( cmd_line )
            subprocess.call(cmd_line, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif k == 'SE100':
            os.system(fr'{opt.client} {opt.output} 101 {d["fov_C"]} {d["fov_R"]} -S -N {report} -E 1')
        elif k == 'PE100':
            os.system(fr'{opt.client} {opt.output} 202 {d["fov_C"]} {d["fov_R"]} -S -N {report} -P 101 -E 3')
        elif k == 'PE100_barcode':
            os.system(fr'{opt.client} {opt.output} 212 {d["fov_C"]} {d["fov_R"]} -S -N {report} -P 101 -E 3 -B {opt.barcode}')
        # summaryReport = fr'./basecall/OutputFq/{report}/L01/{report}_L01.summaryReport.html'
        summaryReport = fr'{opt.basecall_dir}/OutputFq/{report}/L01/{report}_L01.summaryReport.html'
        soup = BeautifulSoup(open(summaryReport, encoding='utf-8'), 'html.parser')
        aa = soup.find_all('script')[1]
        text = aa.text
        ESR = re.findall(r'.+?ESR(.+?)]', text, re.VERBOSE | re.DOTALL)[0]
        ESR = ESR[7:]
        ESR = ESR.replace("'", "")
        # opt.logger.info(fr'{i} {k}_filter_{opt.writeFqFilter} ESR: {ESR}%')
        if k == 'PE100_barcode':
            SplitRate = re.findall(r'.+?SplitRate(.+?)]', text, re.VERBOSE | re.DOTALL)[0]
            SplitRate = SplitRate[7:]
            SplitRate = SplitRate.replace("'", "")
            # opt.logger.info(fr'{i} {k}_filter_{opt.writeFqFilter} SplitRate: {SplitRate}%')
        MappingRate = re.findall(r'.+?MappingRate(.+?)]', text, re.VERBOSE | re.DOTALL)[0]
        MappingRate = MappingRate[7:]
        MappingRate = MappingRate.replace("'", "")
        # opt.logger.info(fr'{i} {k}_filter_{opt.writeFqFilter} MappingRate: {MappingRate}%')
        AvgErrorRate = re.findall(r'.+?AvgErrorRate(.+?)]', text, re.VERBOSE | re.DOTALL)[0]
        AvgErrorRate = AvgErrorRate[7:]
        AvgErrorRate = AvgErrorRate.replace("'", "")
        # opt.logger.info(fr'{i} {k}_filter_{opt.writeFqFilter} AvgErrorRate: {AvgErrorRate}%')
        Q30 = re.findall(r'.+?Q30(.+?)]', text, re.VERBOSE | re.DOTALL)[0]
        Q30 = Q30[7:]
        Q30 = Q30.replace("'", "")
        # opt.logger.info(fr'{i} {k}_filter_{opt.writeFqFilter} Q30: {Q30}%')
    # os.system(f'taskkill /im {process.pid} -f')
    result_dict = {'ESR':ESR, 'MappingRate': MappingRate, 'AvgErrorRate': AvgErrorRate, 'Q30': Q30}
    return result_dict