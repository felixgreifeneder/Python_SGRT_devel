__author__ = 'eodc'

# this routine reads a list of files to process and returns a list of not processed files

import numpy as np
import csv
import untangle
import sgrt.common.recursive_filesearch

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