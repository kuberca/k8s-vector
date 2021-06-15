#!/usr/bin/env python

# given build log file and service log file, split the two into each test run based on namespace
# build log contains if the test run is success or not, we'll put the service log splits into ok/failed directories
# for each splited service log, also create an additional file only contain the timestamp and the 
# time it takes from last log to current one

import json
import logging
import os
import subprocess
import sys
import time
import argparse
import re
import gzip
#from datetime import datetime
import pandas as pd

from os import listdir
from os.path import isfile, isdir, join

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


print("failued cases", failedCases)
# construct a regex contains all the namespaces, to filter out logs from service log
# so if the log does not contain the namespace, the the log is actually not selected out, this could be an issue
# the namespace is like node-port-7890, or node-port-1234-7890, because there are many namespaces
# we only concatinate the alphabet part of all of them, then add a digit suffix, to speed up the match process
nsprefix={}
for n in namespaces:
    sp = n.split("-")
    newarr = [st for st in sp if not st.isdecimal()]
    nsprefix["-".join(newarr)]=""

# to match namespace only
# the [^a-z] at the very begining is to avoid it match to some substring like
# pod="provisioning-4020/pod-subpath-test-dynamicpv-629d"
# without it, above will map to namespace of 'pv-629' 
prefix = "(" + "|".join(nsprefix) + ")"
sufix = "((-\d{1,4}))"
rep = re.compile(".*[^a-z](" + prefix + sufix + ").*")


# to match namespace/obj:  .*((volumemode)((-\d{1,4})*)(\/?([-a-z0-9])*)?).*
# example: volumemode-5139-8151/csi-hostpathplugin

matchFullName = True

if matchFullName:
    sufixFull = "(-\d{1,4})+"
    objName = "(\/([-a-z0-9])+)"
    repFull = re.compile(".*[^a-z](" + prefix + sufixFull + objName + ").*")

# above regex will match some urls like: GET:https://kind-control-plane:6443/api/v1/namespaces/webhook-9749/pods/to-be-attached-pod
# so we filter out these ones 
apiRes = ["secrets","configmaps","pods","serviceaccounts","persistentvolumes","events","csi"]
apiMap={}
for r in apiRes:
    apiMap["/"+r]=True

# results: map[namespace] => (map[objName]=>[lines])
# if not matchFullName, then objName == namespace
# keep the two level dict to make it easier for later code of writing output

# for log lines not matched with any namespace
NoNS="no-ns"

results[NoNS]=[]
bf = open(service_log)
for line in bf:
    m = rep.match(line)
    if m:
        ns = m.group(1)

        if ns not in results:
            results[ns] = {}

        # default use namespace as obj for not matchFullName
        obj = ns

        # if matchFullname, then lets replace / with -- in the name
        if matchFullName:
            #m1 = repFull.match(line)
            m1 = re.match(repFull, line)
            if m1:
                obj = m1.group(1)
                resource = m1.group(4)

                # matching the lines like GET:https://kind-control-plane:6443/api/v1/namespaces/webhook-9749/pods/to-be-attached-pod
                if resource in apiMap:
                    obj = ns

                obj = obj.replace("/", "--")
                # print("ns is" + ns, "obj is " + obj)


        if obj in results[ns]:
            results[ns][obj].append(line)
        else:
            results[ns][obj] = [line]       

    else:
        results[NoNS].append(line)



# write out the no-ns lines
with open(service_log+".nons", "w") as f:
    f.writelines(results[NoNS])

# timeseq keep the time of each log, and the time it takes from last log to current log
timeseq = []

# just use 19:08:25.124718 as format
timestr = re.compile(".*(\d\d:\d\d:\d\d\.\d+).*") 


for ns in results:
    if ns == NoNS:
        continue

    timeseq = []
    lastTime = None

    if ns in failedCases:
        outdir = join(service_log+".err")
    else:
        outdir = join(service_log+".ok")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # print("\n"+ns+"\n")
    # print(results[ns])

    for obj in results[ns]:

        with open(join(outdir, obj), "w") as f:
            f.writelines(results[ns][obj])

        for line in results[ns][obj]:
            # kcm log timestamp format:         2021-04-02T19:08:25.125001456Z stderr F I0402 19:08:25.124718       1 namespace_controller.go:185]
            # kubelet log timestamp format:     Mar 26 05:50:45 kind-worker kubelet[245]: I0326 05:50:45.758222     245 factory.go:220]
            # containerd log format:            Mar 26 05:48:31 kind-worker containerd[174]: time="2021-03-26T05:48:31.039455687Z" level=info
            # date_time_obj = pd.to_datetime(timestr, format='%Y-%m-%dT%H:%M:%S.%fZ')

            m = timestr.match(line)
            if m:
                try:
                    date_time_obj = pd.to_datetime(m.group(1), format='%H:%M:%S.%f')
                except:
                    print("conver to date time exception")
                    print("string", m.group(1))
                    print("line", line)
                    continue
            else:
                continue
            if date_time_obj: 
                if len(timeseq) == 0:
                    lastTime = date_time_obj
                    timeseq.append((m.group(1), 0))
                else:
                    diff = (date_time_obj-lastTime).total_seconds()
                    if diff < 0.0001:
                        diff = 0
                    lastTime = date_time_obj
                    timeseq.append((m.group(1), diff))

        with open(join(outdir, obj + ".time"), 'w') as pf:
            pf.write("\n".join(map(str, timeseq)))  




