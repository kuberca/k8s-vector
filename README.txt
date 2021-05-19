Hello!

data is at:  https://github.com/kuberca/k8s-vector/releases/download/0.0.1/kube-controller-manager.tar.gz

split-file.py to split the build log and service log into multiple files based on the test case (namespace)

rmtime.sh remove time info in the service log file to get only the log message for training

clean.py to remove all the special chars like :;\/

# pipeline to  train the language model

1. combine all service logs
2. run rmtime.sh
3. run clean.py
4. run fasttext -skipgram -input test.log -output model/k8s


# generate vectors for each line of log

cat log.file | fasttext print-sentence-vectors model/k8s.bin

# related projects

https://github.com/logpai/logparser             A collection of log parser, includes drain

https://github.com/wuyifan18/DeepLog            deeplog

https://github.com/donglee-afar/logdeep         couple models try to reproduce deeplog/loganormaly/robustlog

https://github.com/IBM/Drain3                   For log parsing

https://github.com/NetManAIOps/Log2Vec          use semantic embedding (synonym, antonym) 

https://github.com/NetManAIOps/LogParse         Turn log parse to a word classification problem, to have an adaptive parsing method

