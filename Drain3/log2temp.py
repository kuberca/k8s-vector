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
if not os.path.isdir(in_log_dir):
    logger.info(f"dir not exist {in_log_dir}")
    quit()

template_miner = TemplateMiner(persistence)

for in_log_file in os.listdir(in_log_dir):
    # print("processing file", in_log_file)
    line_count = 0
    start_time = time.time()
    batch_start_time = start_time
    batch_size = 200
    ids=[]
    with open(in_log_dir + "/" + in_log_file) as f:
        for line in f:
            line = line.rstrip()
            #line = line.partition(": ")[2]

            #skip the line contains object info or patchBytes
            if "ManagedFields" in line or "ObjectMeta" in line or "patchBytes=[" in line :
                continue
            result = template_miner.add_log_message(line)
            ids.append(result["cluster_id"])
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
    with open(in_log_dir + "../template/" + in_log_file + ".out", "w") as fw:
        fw.write(" ".join(map(str, ids)) + "\n")


time_took = time.time() - start_time
rate = line_count / time_took

# sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)

sorted_clusters = template_miner.drain.clusters


with open(in_log_dir + "../drain.out", "w") as f: 
    f.write(f"--- Done processing file. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
            f"{len(template_miner.drain.clusters)} clusters\n")
    for cluster in sorted_clusters:
        f.write(str(cluster)+"\n")

with open(in_log_dir + "../templates.txt", "w") as f: 
    for cluster in sorted_clusters:
        f.write(cluster.get_template()+"\n")

# print("Prefix Tree:")
# template_miner.drain.print_tree()

# template_miner.profiler.report(0)


