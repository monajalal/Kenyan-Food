import json
import urllib
import requests
import shutil
from json import JSONDecoder
from functools import partial
import sys
import os
import itertools
from os import path


def json_parse(fileobj, decoder=JSONDecoder(), buffersize=2048):
    buffer = ''
    for chunk in iter(partial(fileobj.read, buffersize), ''):
        buffer += chunk
        while buffer:
            try:
                result, index = decoder.raw_decode(buffer)
                yield result
                buffer = buffer[index:].lstrip()
            except ValueError:
                # Not enough data to decode, read more
                break


def download_image(p, fn, url, i):
    global total_downloads
    try:
        print("Downloading = " + str(total_downloads) + "\n url = " + url + "\n")
        resp = requests.get(url, stream=True)
        local_file = open(p + filename + '.jpg', 'wb')
        resp.raw.decode_content = True
        shutil.copyfileobj(resp.raw, local_file)
        local_file.close()
        total_downloads +=1
        print("Download success!")
        del resp
    except requests.exceptions.RequestException as e:
        return

def download_video(p, fn, url, i):
    global total_downloads
    if url == None or url == "":
        return
    try:
        print("Downloading = " + str(total_downloads) + "\n url = " + url + "\n")
        resp = requests.get(url, stream=True)
        local_file = open(p + filename + '.mp4', 'wb')
        resp.raw.decode_content = True
        shutil.copyfileobj(resp.raw, local_file)
        local_file.close()
        total_downloads +=1
        print("Download success!")
        del resp
    except requests.exceptions.RequestException as e:
        return


skip = 0
if(len(sys.argv) > 2 and (sys.argv[2] is not None)):
    skip = int(sys.argv[2])
total_downloads = skip
print(str(skip))
current_dir = str(os.getcwd()) + '/'
foodpath = current_dir + 'img_' + (str(sys.argv[1])[25:-5]) + '/food/'
nonfoodpath = current_dir + 'img_' + (str(sys.argv[1])[25:-5]) + '/nonfood/'
videopath = current_dir + 'video_/'

if not path.isdir("img_") :
    try:
        os.mkdir(foodpath[:-5])
    except OSError:
        print("Creation of the directory failed")
    else:
        print("Successfully created the directory")
    try:
        os.mkdir(foodpath[:-1])
    except OSError:
        print("Creation of the directory failed")
    else:
        print("Successfully created the directory")
    try:
        os.mkdir(nonfoodpath[:-1])
    except OSError:
        print("Creation of the directory failed")
    else:
        print("Successfully created the directory")
    try:
        os.mkdir(videopath[:-1])
    except OSError:
        print("Creation of video directory failed")
    else:
        print("Successfully created the video directory")
    



Wdata = []
with open(str(sys.argv[1])) as f:
    for line in itertools.islice(f, skip, None):
        ##print(len(Wdata))
        ##print(line)
        try:
            Wdata.append(json.loads(line))
        except ValueError:
            continue
    ##with open('temp_loc_output_-1.57085_0_36.6_38.3.json', 'r') as infh:
    i = 0
    for data in Wdata:
        image_url = data["output_content"]["graphql"]["shortcode_media"]["display_url"]
        if data["output_content"]["graphql"]["shortcode_media"]["is_video"] == True:
            video_url = data["output_content"]["graphql"]["shortcode_media"]["video_url"]
        path = nonfoodpath
        filename = data["output_content"]["graphql"]["shortcode_media"]["id"]
        if ("edge_sidecar_to_children" in data["output_content"]["graphql"]["shortcode_media"]):
            for child in data["output_content"]["graphql"]["shortcode_media"]["edge_sidecar_to_children"]["edges"]:
                image_url = child["node"]["display_url"]
                if child["node"]["is_video"] == True:
                    video_url = child["node"]["video_url"]
                filename = child["node"]["id"]

                if ("accessibility_caption" in child["node"]):
                    if (child["node"]["accessibility_caption"] != None):
                        if ("food" in child["node"]["accessibility_caption"]):
                            path = foodpath
                            ##print(filename)
                            ##print(child["node"]["accessibility_caption"])
                            ##print(data["graphql"]["shortcode_media"]["edge_media_to_caption"]["edges"][0]["node"]["text"])
                        else:
                            path = nonfoodpath
                    else:
                        path = nonfoodpath
                else:
                    path = nonfoodpath

                if (child["node"]["is_video"] != True):
                    i += 1
                    # print(i)
                    download_image(path, filename, image_url, i)
                else:
                    download_video(videopath, filename, video_url, i)
        else:
            # if (data["graphql"]["shortcode_media"]["is_video"] == 'true'):

            if ("accessibility_caption" in data["output_content"]["graphql"]["shortcode_media"]):
                if (data["output_content"]["graphql"]["shortcode_media"]["accessibility_caption"] != None):
                    if ("food" in data["output_content"]["graphql"]["shortcode_media"]["accessibility_caption"]):
                        path = foodpath
                        ##print(filename)
                        ##print(data["graphql"]["shortcode_media"]["accessibility_caption"])
                        ##print(data["graphql"]["shortcode_media"]["edge_media_to_caption"]["edges"][0]["node"]["text"])
                    else:
                        path = nonfoodpath
                else:
                    path = nonfoodpath
            else:
                path = nonfoodpath

            if (data["output_content"]["graphql"]["shortcode_media"]["is_video"] != True):
                i += 1
                # print(i)
                download_image(path, filename, image_url, i)
            else:
                download_video(videopath, filename, video_url, i)
