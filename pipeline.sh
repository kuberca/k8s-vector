#!/bin/bash

# 1. split the file based on build log
# output: $i-1/2.log.ok/  
items="kl1 kl2 kl3 kl4 kl5 kl6 kl7 kl8 kl9 kl10"

#for i in $items; do 
#    python split-file.py /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-build.log /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-1.log; 
#    python split-file.py /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-build.log /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-2.log;
#done


# 2. python convert the
# cd Drain3
# for i in $items; do 
#     python log2temp_time.py /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-1.log.ok > /tmp/t.t; 
#     python log2temp_time.py /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-2.log.ok > /tmp/t.t; 
#     python log2temp_time.py /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-1.log.err > /tmp/t.t; 
#     python log2temp_time.py /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-2.log.err > /tmp/t.t; 
# done
# cd ../

# 3 combine all templates into all.ok, test.ok, test.err
rm all.ok train.ok test.ok test.err

for i in $items; do 
    cat /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-1.log.ok.template/* >> all.ok
    cat /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-2.log.ok.template/* >> all.ok

    # cat /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-1.log.err.template/* >> test.err
    # cat /Users/junzhou/code/rca/data/pk8s/kubelet/alllog/$i-2.log.err.template/* >> test.err
done

awk 'NR%10!=0' all.ok > train.ok
awk 'NR%10==0' all.ok > test.ok
cat alllog/realerr.template/* >> test.err

# 4 generate vector

# cat alllog/drain.out | awk -F "," '{print $3}' > alllog/templates.txt
# cat alllog/templates.txt | fasttext print-sentence-vectors model/k8s.bin > alllog/templates.txt.vec


# python sim.py alllog/templates.txt.vec


