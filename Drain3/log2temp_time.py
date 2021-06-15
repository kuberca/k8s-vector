"""
Description : Example of using Drain3 to process a real world file
Author      : David Ohana
Author_email: david.ohana@ibm.com
License     : MIT
"""
import json
import logging
import os
import subprocess
import sys
import time
import argparse
import re
import pandas as pd

from drain3 import TemplateMiner

from drain3.file_persistence import FilePersistence

persistence = FilePersistence("drain3_state.bin")

parser = argparse.ArgumentParser(description='Process the log file in dir.')
parser.add_argument('dir', metavar='dir', type=str, 
                    help='file directory')

args = parser.parse_args()

print(args.dir)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


in_log_dir = args.dir
if in_log_dir.endswith("/"):
    in_log_dir = in_log_dir[:-1]

if not os.path.isdir(in_log_dir):
    logger.info(f"dir not exist {in_log_dir}")
    quit()

output_dir = in_log_dir + ".template"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


template_miner = TemplateMiner(persistence)

# logline regex to extract timestamp and actual content
# time 19:08:25.124718 as format
# example log: Mar 26 17:14:02 kind-worker kubelet[246]: I0326 17:14:02.126812     246 volume_manager.go:404] "All volumes are attached and mounted for pod" pod="provisioning-8300/hostexec-kind-worker-bqvpk"
logregex = re.compile(".*(\d\d:\d\d:\d\d\.\d+).* [^ ]+\.go:\d+] (.*)$")

# skipWords = ["ManagedFields", "ObjectMeta", "patchBytes=[", "PLEG: Write status", "Event occurred", "httplog.go", "SyncLoop"]
skipWords = ["ManagedFields", "ObjectMeta", "patchBytes=[", "PLEG: Write status", "httplog.go", "SyncLoop", "InitContainerStatuses", "ResourceRequirements", "Error syncing pod", "Event occurred", "but some containers have not been cleaned up"]

for in_log_file in os.listdir(in_log_dir):
    # only process those files with actual resource name
    # like: kubectl-9089--failure-1
    # files like kubectl-9089  doesn't have any resource names
    if "--" not in in_log_file:
        continue

    line_count = 0
    start_time = time.time()
    batch_start_time = start_time
    batch_size = 200
    ids=[]
    lastTime = None
    with open(in_log_dir + "/" + in_log_file) as f:
        for line in f:
            line = line.rstrip()
            #line = line.partition(": ")[2]

            #for the line contains object info or patchBytes, we only keep the part of the line which include the message and one param
            skip = False
            for i in skipWords:
                if i in line:
                    skip = True
                    break
            if skip:
                cnt = 0
                charidx = []
                for i, ltr in enumerate(line):
                    if ltr == '"':
                        # dont want lines too long
                        if i > 500:
                            break
                        charidx.append(i)
                        cnt += 1

                    if cnt == 4:
                        break
                if len(charidx) > 0:    
                    line = line[:charidx[-1]]
            
            # skip the lines with too many :, most likely are lines print out the object content
            # allifx = [i for i, ltr in enumerate(line) if ltr == ":"]
            # if len(allifx) > 20:
            #     line = line[:200]

            m = logregex.match(line)
            if not m:
                continue

            timestamp = m.group(1)
            content = m.group(2)
            date_time_obj = pd.to_datetime(timestamp, format='%H:%M:%S.%f')
            diff = 0
            if len(ids) == 0:
                lastTime = date_time_obj
            else:
                diff = (date_time_obj-lastTime).total_seconds()
                if diff < 0.0001:
                    diff = 0
                lastTime = date_time_obj

            result = template_miner.add_log_message(content)

            # keep a tuple (eventId, timediff) for each log line
            out = "{},{}"
            ids.append(out.format(result["cluster_id"],diff))

            line_count += 1
            if line_count % batch_size == 0:
                time_took = time.time() - batch_start_time
                rate = batch_size / time_took
                logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                            f"{len(template_miner.drain.clusters)} clusters so far.")
                batch_start_time = time.time()
            if result["change_type"] == "cluster_template_changed":
                result_json = json.dumps(result)
                # logger.info(f"Input ({line_count}): " + line + "\n")
                # logger.info("Result: " + result_json + "\n")


    # print("file processed.", in_log_file, ids)
    if len(ids) > 0:
        with open(os.path.join(output_dir, in_log_file), "w") as fw:
            fw.write(" ".join(ids))
            fw.write("\n")


time_took = time.time() - start_time
rate = line_count / time_took

# sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)

sorted_clusters = template_miner.drain.clusters


with open(in_log_dir + "/../drain.out", "w") as f: 
    # f.write(f"--- Done processing file. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
            # f"{len(template_miner.drain.clusters)} clusters\n")
    for cluster in sorted_clusters:
        f.write("{},{},{}\n".format(cluster.cluster_id-1, cluster.size, cluster.get_template()))
        # f.write(str(cluster)+"\n")

with open(in_log_dir + "/../templates.txt", "w") as f: 
    for cluster in sorted_clusters:
        f.write(cluster.get_template()+"\n")

# print("Prefix Tree:")
# template_miner.drain.print_tree()

# template_miner.profiler.report(0)


