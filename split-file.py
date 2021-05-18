#!/usr/bin/env python

import json
import logging
import os
import subprocess
import sys
import time
import argparse
import re
import gzip

from os import listdir
from os.path import isfile, isdir, join


# given build log file and service log file, split the two into each test run based on namespace
# build log contains if the test run is success or not, we'll put the service log splits into ok/failed directories
parser = argparse.ArgumentParser(description='split logs for each test case.')
parser.add_argument('build_log', metavar='build_log', type=str, help='build log file')
parser.add_argument('service_log', metavar='service_log', type=str, help='service log file')

args = parser.parse_args()

build_log = args.build_log
service_log = args.service_log

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

build_log_out_dir_ok = build_log + ".ok"
build_log_out_dir_err = build_log + ".err"
if not isdir(build_log_out_dir_ok):
    os.mkdir(build_log_out_dir_ok)
if not isdir(build_log_out_dir_err):
    os.mkdir(build_log_out_dir_err)

if not os.path.isfile(build_log):
    logger.info(f"build log file not exist {build_log}")
    quit()

#############################################################
# split the build log based on regex pattens
#############################################################
# summary = re.compile(".*Passed.*Failed.*Pending.*Skipped.*")
# start 
#       [BeforeEach]
# end:  
#       STEP: Destroying namespace "container-runtime-7928" for this suite.
#           continue to next [BeforeEach], set as end
#           extract namespace
#           write start/end to separte file
#           continue
#   OR
#       [SKIPPING]  
#           discard tmp data
#           continue  
start = re.compile("^(.\d{4} \d{2}:\d{2}:\d{2}\.\d{3}\] )?\[BeforeEach\].*")                                    # start of test case
end = re.compile(".* Destroying namespace \"(.*)\" for this suite.*")                                           # end of test case
endNamespace = re.compile(".*Waiting for namespaces \[(.*)\] to vanish.*")                                      # namespace in end of test case
skip = re.compile(".*\[SKIPPING\].*")                                                                           # test case skipped
summary = re.compile(".*m(\d+) Passed.* \| .*m(\d+) Failed.* \| .*m(\d+) Pending.* \| .*m(\d+) Skipped.*")      # summary of the whole build
failure = re.compile(".*Failure \[\d+\.\d+ .*\].*")                                                             # failure information
namespaces = []                                                                                                 # namespace is used to split the logs because each test case is using a new namespace
results = {}
failedCases = {}

with open(build_log) as bf:
    store = ""                                  # to keep logs for current test case
    namespace = ""                              # namespace used for current test case
    matching = False                            # when true, means already found a start
    matchingEnd = False                         # when true, means found an end or summary
    failed = False

    for line in bf:
        if start.match(line):
            if matchingEnd:
                # write the data to a file with namespace as the file name
                outdir = build_log_out_dir_ok
                if failed:
                    failedCases[namespace]=True
                    outdir = build_log_out_dir_err
                with open(join(outdir, namespace), 'w') as pf:
                    pf.write(store)
                # print("found match for namespace", namespace, "failed: ", failed)
                store = ""
                namespace = ""
                matching = False
                matchingEnd = False
                failed = False

            matching = True
            store = store + line
        elif summary.match(line):
            # build_log parsing finishing, write out the data for last test case
            outdir = build_log_out_dir_ok
            if failed:
                failedCases[namespace]=True
                outdir = build_log_out_dir_err
            with open(join(outdir, namespace), 'w') as pf:
                pf.write(store)
            print("found match for namespace at the end", namespace, "failed: ", failed)
            break
            store = ""
            namespace = ""
            matching = False
            failed = False
        elif skip.match(line):
            # test case skipped
            matching = False
            store = ""
            continue
        elif end.match(line) or endNamespace.match(line):
            store += line
            # if already found end, and another end coming, means there are multiple ns to be destroyed, we just pick the first one which should have the common prefix
            if matchingEnd:    
                continue

            m = end.match(line)
            if not m:
                m = endNamespace.match(line)
            namespace = m.group(1)
            namespaces.append(namespace)

            # now we found the destroying line, continue to the next BeforeEach, then write the data
            matchingEnd = True
            continue 
        else:
            if matching:
                store += line
            if matchingEnd:
                if failure.match(line):
                    failed = True


# construct a regex contains all the namespaces, to filter out logs from service log
# so if the log does not contain the namespace, the the log is actually not selected out, this could be an issue
# the namespace is like node-port-7890, or node-port-1234-7890, because there are many namespaces
# we only concatinate the alphabet part of all of them, then add a digit suffix, to speed up the match process
nsprefix={}
for n in namespaces:
    sp = n.split("-")
    newarr = [st for st in sp if not st.isdecimal()]
    nsprefix["-".join(newarr)]=""

prefix = "(" + "|".join(nsprefix) + ")"
sufix = "((-\d{1,4}))"
rep = re.compile(".*(" + prefix + sufix + ").*")

# teststr = "I0331 18:34:44.120761      10 eventhandlers.go:279] \"Delete event for scheduled pod\" pod=\"csi-mock-volumes-1570-6273/csi-mockplugin-0\""
# m = rep.match(teststr)
# if m:
#     print(m.group(1))


# art_dir = join(log_dir,"artifacts")


# dirs = [f for f in listdir(art_dir) if isdir(join(art_dir, f))]
# for d in dirs:
#     dir=join(art_dir, d)
#     files = [f for f in listdir(dir) if isfile(join(dir, f))]

# for root, dirs, files in os.walk(art_dir):
    # for file in files:
    
# if not "log" in file:
#     continue
# 

# if file.endswith(".log"):
#     bf = open(join(root,file))
# else:
#     bf = gzip.open(join(root, file), "rt")
#     try: 
#         bf.readline()
#     except gzip.BadGzipFile:
#         print(file, "is not a gzip file")
#         bf = open(join(root,file))

for n in namespaces:
     results[n] = []

bf = open(service_log)
for line in bf:
    m = rep.match(line)
    if m:
        ns = m.group(1)
        # this ns may not be the exact one
        if ns in results:
            results[ns].append(line)


for n in namespaces:
    if results[n]:
        if n in failedCases:
            outdir = join(service_log+".err")
        else:
            outdir = join(service_log+".ok")
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        with open(join(outdir, n), "w") as f:
            f.writelines(results[n])


