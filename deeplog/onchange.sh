#!/bin/bash

run_rsync() {
 rsync -avt .  root@jump:/root/rca/k8s/code
}

fswatch -o . | while read f; do run_rsync; done
