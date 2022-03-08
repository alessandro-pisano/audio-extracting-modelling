# Importing necessary Packages
from bs4 import BeautifulSoup
from multiprocessing import Pool

import http.client
import json
import io, os
import requests
import time
import wave
import wavio
import zipfile

import numpy as np
import pandas as pd
import speech_recognition as sr
import soundfile as sf


def download_file(url, folder):
    """Function to download and save file"""
    local_filename = url.split('/')[-1]
    r = requests.get(url)
    path = folder + "/" + local_filename
    f = open(path, 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024): 
        if chunk:
            f.write(chunk)
    f.close()
    return 

def get_html(url):
    """Function to read html page and get URL"""
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("a")
    files = []
    for audio in results:
        files.append(audio.text.strip())
    return files

def download_audio(url, files, folder):
    """Function to download all audio from TalkBank"""
    os.makedirs(folder, exist_ok=True)
    for file in files:
        if "/" in file:
            new_url = url + "/" + file
            print('found to iterate in', new_url)
            files = get_html(new_url)
            print(files)
            download_audio(new_url, files, folder)
        if ".wav" in file:
            down = url + "/" + file
            print(f"Downloading {down}")
            download_file(down, folder)
            time.sleep(0.5)

def down_sent(surveys_all, df):
    """Function to download all surveys"""
    problems_fold = []
    for survey_id in surveys_all:
        folder = df.loc[surveys_all.index(survey_id),"Folder Call"]
        try:
            conn = http.client.HTTPSConnection("unibocconi.qualtrics.com")
            payload = '{"format":"csv"}'
            conn.request("POST", "/API/v3/surveys/%s/export-responses/" % survey_id, payload, headers=headers)
            res = conn.getresponse()
            data = res.read()
            progress_id = json.loads(data)["result"]["progressId"]

            conn = http.client.HTTPSConnection("unibocconi.qualtrics.com")
            conn.request("GET", "/API/v3/surveys/%s/v2-api-export-responses/%s" % (survey_id, progress_id), headers=headers)
            res = conn.getresponse()
            data = res.read()
            file_id = json.loads(data)["result"]["fileId"]

            conn = http.client.HTTPSConnection("unibocconi.qualtrics.com")
            conn.request("GET", "/API/v3/surveys/%s/v2-api-export-responses/%s/file" % (survey_id, file_id), headers=headers)
            res = conn.getresponse()
            data = res.read()
            zipfile.ZipFile(io.BytesIO(data)).extractall("Results_Folder/")
        except:
            print(f"problem with {folder}")
            problems_fold.append(survey_id)
            
    print("len is", len(problems_fold))
            
    if len(problems_fold)==0:
        print("done")
    else:
        print("Iterating for", len(problems_fold), "files")
        down_sent(problems_fold, df)
        
    return problems_fold

def start_conversion(path, lang='en-US', r = sr.Recognizer()): 
    """Transcribing audio file"""
    with sr.AudioFile(path) as source:
        print(f'Fetching File: {path}')
        audio_file = r.record(source)
        return r.recognize_google(audio_file, language=lang)

def millisec(millis):
    """Millisec function"""
    millis = int(millis)
    seconds, mil = divmod((millis/1000)%60,1)
    seconds = str(round(seconds)).zfill(2)
    mil = str(round(mil*100)).zfill(2)
    minutes = str(round((millis/(1000*60))%60)).zfill(2)
    return ("{}:{}:{}".format(minutes, seconds, mil))

def creating_transcripts(folder):
    """Generating Transcripts"""
    print(folder)
    new_dir = folder + "/Audio_Speech/"
    os.makedirs(folder + "/Transcripts", exist_ok=True)
    for audio_fold in sorted(os.listdir(new_dir)):
        print(f"Transcripting Audio: {audio_fold}")
        file_tr = ""
        trans_dir = folder + "/Transcripts/" + audio_fold  + "_tr.txt"
        with open(trans_dir, "w+") as transcript_file:
            
            for part in sorted(os.listdir(new_dir + audio_fold), 
                               key=lambda x: (int(x.split('_')[1]), 
                                              int(x.split('_')[2]))):
                file_audio = new_dir + audio_fold + "/" + part
                name_f = part.split("_")
                try:
                    transcript = start_conversion(file_audio)
                except:
                    print("\nImpossible\n")
                    transcript = "*sounds*"
                #print(transcript)
                file_tr += str(millisec(name_f[1]) + "\t" + millisec(name_f[2]) \
                               + "\t" + name_f[3][0] + ":\t" + transcript + "\n")
            transcript_file.write(file_tr)


def split_data(file, train_dim=0.80, dev_dim=0.10, test_dim = 0.1):
    """Split the data in training, development and test set"""
    folders = file["folder"].unique()
    length = len(folders)
    train_ind = 0
    dev_ind = int(round(length * train_dim))
    test_ind = int(round(dev_ind + length * dev_dim))
    index = []

    for i in [0, dev_ind, test_ind]:
        index.append(file[file['folder'] == folders[i]].index[0])

    db_train = file[:index[1]]
    db_dev = file[index[1]: index[2]]
    db_test = file[index[2]:]
    
    return db_train.copy(), db_dev.copy(), db_test.copy()