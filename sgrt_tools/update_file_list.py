__author__ = 'eodc'

# this routine reads a list of files to process and returns a list of not processed files

import numpy as np
import csv
import untangle
import sgrt.common.recursive_filesearch
import os.path


def remove_processed_from_list(filelist, logfile, outpath):

    # read original file list

    with open (filelist, "rb") as f:
        file_list_old = f.readlines()

    # read log-file
    logobj = untangle.parse(logfile)
    processed = logobj.root.process_log.list_of_processed_files.cdata.encode().strip()
    processed = processed.split(", ")

    file_list_new = list()

    for listitem in file_list_old:
        exists = 0
        for pfile in processed:
            if os.path.basename(listitem.strip()) == pfile:
                exists = 1

        if exists == 0:
            file_list_new.append(listitem.strip())

    # write results
    f = open(outpath, "wb")
    for x in file_list_new: f.write(x + '\n')
    #f.writelines(filelist)
    f.close()


def update(inpath, logfilepath, outpath):

    # read original
    with open(inpath, "rb") as f:
        reader = csv.reader(f)
        filelist_old = list(reader)

    logobj = untangle.parse(logfilepath)
    log = {'notprocessed': logobj.root.process_log.list_of_not_processed_files.cdata.encode().strip()}
    log = log['notprocessed'].split(", ")

    filelist_new = sgrt.common.recursive_filesearch.search_file("/eodc/sentinel/pub/ESA/Sentinel_1A_CSAR/IW/GRDH/datasets/", log)

    # filelist_new = []
    # for fname in log:
    #     for fpath in filelist_old:
    #         fs = fpath[0].find(fname)
    #         if fs > -1:
    #             filelist_new.append(fpath[0])

    with open(outpath, "wb") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, lineterminator='\n', delimiter=",")
        writer.writerow(filelist_new)


def scihub2list(inpath, outpath):

    # get the list of file names from the sciHub search results
    logobj = untangle.parse(inpath)
    entrylist = logobj.feed.entry

    filenames = list()
    for obj in entrylist:
        filenames.append(obj.title.cdata.encode().strip())

    pathlist = list()
    for file in filenames:
        year = file[17:21]
        month = file[21:23]
        day = file[23:25]
        pathlist.append("/eodc/pub/copernicus.eu/s1a_csar_grdh_iw/" + year + '/' + month + '/' + day + '/')

    # get the full file names
    filelist = list()
    for i in range(len(filenames)):
        result = sgrt.common.recursive_filesearch.search_file(pathlist[i], filenames[i]+'.zip')
        if len(result) != 0:
            filelist.append(result[0])

    # write results
    f = open(outpath, "wb")
    for x in filelist: f.write(x + '\n')
    #f.writelines(filelist)
    f.close()

    #with open(outpath, "wb") as f:
    #    writer = csv.writer(f, quoting=csv.QUOTE_NONE, lineterminator='\n', delimiter=',')
    #    writer.writerow(filelist)