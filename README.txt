Hello!

split-file.py to split the build log and service log into multiple files based on the test case (namespace)

rmtime.sh remove time info in the service log file to get only the log message for training

clean.py to remove all the special chars like :;\/

# pipeline to  train the language model
1. combine all success test case service logs
2. run rmtime.sh
3. run clean.py
4. run fasttext -skipgram -input test.log -output model/k8s


# generate vectors for each line of log
cat log.file | fasttext print-sentence-vectors model/k8s.bin
