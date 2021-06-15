/Users/junzhou/code/rca/data/pk8s/kubelet/alllog/kl4-1.log.ok
Starting Drain3 template miner
Loading configuration from drain3.ini
Checking for saved state
Saved state not found
content:  "Receiving a new pod" pod="gc-1230/simpletest.rc-9m9bs"
old template:  Receiving a new pod pod gc-1230/simpletest.rc-2f6d6
new template:  Receiving a new pod pod <*> 

content:  "Added volume to desired state" pod="gc-1230/simpletest.rc-2f6d6" volumeName="kube-api-access-49nvt" volumeSpecName="kube-api-access-49nvt"
old template:  Added volume to desired state pod gc-1230/simpletest.rc-9m9bs volumeName kube-api-access-dx8nr volumeSpecName kube-api-access-dx8nr
new template:  Added volume to desired state pod <*> volumeName <*> volumeSpecName <*> 

content:  "Generating pod status" pod="gc-1230/simpletest.rc-9m9bs"
old template:  Generating pod status pod gc-1230/simpletest.rc-2f6d6
new template:  Generating pod status pod <*> 

content:  "Waiting for volumes to attach and mount for pod" pod="gc-1230/simpletest.rc-9m9bs"
old template:  Waiting for volumes to attach and mount for pod pod gc-1230/simpletest.rc-2f6d6
new template:  Waiting for volumes to attach and mount for pod pod <*> 

content:  pod gc-1230/simpletest.rc-9m9bs volume kube-api-access-dx8nr: performed write of new data to ts data directory: /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~projected/kube-api-access-dx8nr/..2021_03_26_17_07_22.800308465
old template:  pod gc-1230/simpletest.rc-2f6d6 volume kube-api-access-49nvt performed write of new data to ts data directory /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~projected/kube-api-access-49nvt/..2021_03_26_17_07_22.155439962
new template:  pod <*> volume <*> performed write of new data to ts data directory <*> 

content:  "All volumes are attached and mounted for pod" pod="gc-1230/simpletest.rc-9m9bs"
old template:  All volumes are attached and mounted for pod pod gc-1230/simpletest.rc-2f6d6
new template:  All volumes are attached and mounted for pod pod <*> 

content:  "No sandbox for pod can be found. Need to start a new one" pod="gc-1230/simpletest.rc-9m9bs"
old template:  No sandbox for pod can be found. Need to start a new one pod gc-1230/simpletest.rc-2f6d6
new template:  No sandbox for pod can be found. Need to start a new one pod <*> 

content:  "computePodActions got for pod" podActions={KillPod:true CreateSandbox:true SandboxID: Attempt:0 NextInitContainerToStart:nil ContainersToStart:[0] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="gc-1230/simpletest.rc-9m9bs"
old template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID Attempt 0 NextInitContainerToStart nil ContainersToStart 0 ContainersToKill map EphemeralContainersToStart pod gc-1230/simpletest.rc-2f6d6
new template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID Attempt 0 NextInitContainerToStart nil ContainersToStart 0 ContainersToKill map EphemeralContainersToStart pod <*> 

content:  "SyncPod received new pod, will create a sandbox for it" pod="gc-1230/simpletest.rc-9m9bs"
old template:  SyncPod received new pod will create a sandbox for it pod gc-1230/simpletest.rc-2f6d6
new template:  SyncPod received new pod will create a sandbox for it pod <*> 

content:  "Stopping PodSandbox for pod, will start new one" pod="gc-1230/simpletest.rc-9m9bs"
old template:  Stopping PodSandbox for pod will start new one pod gc-1230/simpletest.rc-2f6d6
new template:  Stopping PodSandbox for pod will start new one pod <*> 

content:  "Creating PodSandbox for pod" pod="gc-1230/simpletest.rc-9m9bs"
old template:  Creating PodSandbox for pod pod gc-1230/simpletest.rc-2f6d6
new template:  Creating PodSandbox for pod pod <*> 

content:  "Created PodSandbox for pod" podSandboxID="<SEQ>" pod="gc-1230/simpletest.rc-2f6d6"
old template:  Created PodSandbox for pod podSandboxID <SEQ> pod gc-1230/simpletest.rc-9m9bs
new template:  Created PodSandbox for pod podSandboxID <SEQ> pod <*> 

content:  "Determined the ip for pod after sandbox changed" IPs=[fd00:10:244:1::16c] pod="gc-1230/simpletest.rc-2f6d6"
old template:  Determined the ip for pod after sandbox changed IPs fd00 10 244 1 16d pod gc-1230/simpletest.rc-9m9bs
new template:  Determined the ip for pod after sandbox changed IPs fd00 10 244 1 <*> pod <*> 

content:  "Creating hosts mount for container" pod="gc-1230/simpletest.rc-2f6d6" containerName="nginx" podIPs=[fd00:10:244:1::16c] path=true
old template:  Creating hosts mount for container pod gc-1230/simpletest.rc-9m9bs containerName nginx podIPs fd00 10 244 1 16d path true
new template:  Creating hosts mount for container pod <*> containerName nginx podIPs fd00 10 244 1 <*> path true 

content:  "Event occurred" object="gc-1230/simpletest.rc-2f6d6
old template:  Event occurred object gc-1230/simpletest.rc-9m9bs
new template:  Event occurred object <*> 

content:  "getSandboxIDByPodUID got sandbox IDs for pod" podSandboxID=[<SEQ>] pod="gc-1230/simpletest.rc-2f6d6"
old template:  getSandboxIDByPodUID got sandbox IDs for pod podSandboxID <SEQ> pod gc-1230/simpletest.rc-9m9bs
new template:  getSandboxIDByPodUID got sandbox IDs for pod podSandboxID <SEQ> pod <*> 

content:  "PLEG: Write status" pod="gc-1230/simpletest.rc-2f6d6
old template:  PLEG Write status pod gc-1230/simpletest.rc-9m9bs
new template:  PLEG Write status pod <*> 

content:  "SyncLoop (PLEG): event for pod" pod="gc-1230/simpletest.rc-2f6d6
old template:  SyncLoop PLEG event for pod pod gc-1230/simpletest.rc-9m9bs
new template:  SyncLoop PLEG event for pod pod <*> 

Processing line: 200, rate 1426.6 lines/sec, 25 clusters so far.
content:  "computePodActions got for pod" podActions={KillPod:false CreateSandbox:false SandboxID:<SEQ> Attempt:0 NextInitContainerToStart:nil ContainersToStart:[] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="gc-1230/simpletest.rc-5pk6c"
old template:  computePodActions got for pod podActions KillPod false CreateSandbox false SandboxID <SEQ> Attempt 0 NextInitContainerToStart nil ContainersToStart ContainersToKill map EphemeralContainersToStart pod gc-1230/simpletest.rc-9m9bs
new template:  computePodActions got for pod podActions KillPod false CreateSandbox false SandboxID <SEQ> Attempt 0 NextInitContainerToStart nil ContainersToStart ContainersToKill map EphemeralContainersToStart pod <*> 

content:  pod gc-1230/simpletest.rc-5pk6c volume kube-api-access-dgp8h: no update required for target directory /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~projected/kube-api-access-dgp8h
old template:  pod gc-1230/simpletest.rc-9m9bs volume kube-api-access-dx8nr no update required for target directory /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~projected/kube-api-access-dx8nr
new template:  pod <*> volume <*> no update required for target directory <*> 

content:  "Patch status for pod" pod="gc-1230/simpletest.rc-8l6x4
old template:  Patch status for pod pod gc-1230/simpletest.rc-42h87
new template:  Patch status for pod pod <*> 

content:  "Status for pod updated successfully" pod="gc-1230/simpletest.rc-8l6x4
old template:  Status for pod updated successfully pod gc-1230/simpletest.rc-42h87
new template:  Status for pod updated successfully pod <*> 

content:  "Killing container with a grace period override" pod="gc-1230/simpletest.rc-8t889" podUID=<UID> containerName="nginx" containerID="containerd://<SEQ>" gracePeriod=2
old template:  Killing container with a grace period override pod gc-1230/simpletest.rc-42h87 podUID <UID> containerName nginx containerID containerd //<SEQ> gracePeriod 2
new template:  Killing container with a grace period override pod <*> podUID <UID> containerName nginx containerID containerd //<SEQ> gracePeriod 2 

content:  "Pod has completed execution and should be deleted from the API server" pod="gc-1230/simpletest.rc-4p7x9" syncType="update"
old template:  Pod has completed execution and should be deleted from the API server pod gc-1230/simpletest.rc-42h87 syncType update
new template:  Pod has completed execution and should be deleted from the API server pod <*> syncType update 

content:  "Container exited normally" pod="gc-1230/simpletest.rc-42h87" podUID=<UID> containerName="nginx" containerID="containerd://<SEQ>"
old template:  Container exited normally pod gc-1230/simpletest.rc-5pk6c podUID <UID> containerName nginx containerID containerd //<SEQ>
new template:  Container exited normally pod <*> podUID <UID> containerName nginx containerID containerd //<SEQ> 

content:  "Failed to delete pod" pod="gc-1230/simpletest.rc-6mx44" err="pod not found"
old template:  Failed to delete pod pod gc-1230/simpletest.rc-4p7x9 err pod not found
new template:  Failed to delete pod pod <*> err pod not found 

content:  "Pod does not exist on the server" podUID=<UID> pod="gc-1230/simpletest.rc-2f6d6"
old template:  Pod does not exist on the server podUID <UID> pod gc-1230/simpletest.rc-9m9bs
new template:  Pod does not exist on the server podUID <UID> pod <*> 

content:  "Removing volume from desired state" pod="gc-1230/simpletest.rc-4p7x9" podUID=<UID> volumeName="kube-api-access-ks8lx"
old template:  Removing volume from desired state pod gc-1230/simpletest.rc-8t889 podUID <UID> volumeName kube-api-access-m4jvh
new template:  Removing volume from desired state pod <*> podUID <UID> volumeName <*> 

Processing line: 400, rate 1644.7 lines/sec, 36 clusters so far.
content:  "Creating hosts mount for container" pod="container-lifecycle-hook-7916/pod-handle-http-request" containerName="agnhost-container" podIPs=[fd00:10:244:1::1ad] path=true
old template:  Creating hosts mount for container pod <*> containerName nginx podIPs fd00 10 244 1 <*> path true
new template:  Creating hosts mount for container pod <*> containerName <*> podIPs fd00 10 244 1 <*> path true 

content:  "Killing container with a grace period override" pod="container-lifecycle-hook-7916/pod-handle-http-request" podUID=<UID> containerName="agnhost-container" containerID="containerd://<SEQ>" gracePeriod=2
old template:  Killing container with a grace period override pod <*> podUID <UID> containerName nginx containerID containerd //<SEQ> gracePeriod 2
new template:  Killing container with a grace period override pod <*> podUID <UID> containerName <*> containerID containerd //<SEQ> gracePeriod 2 

content:  "Container exited normally" pod="container-lifecycle-hook-7916/pod-handle-http-request" podUID=<UID> containerName="agnhost-container" containerID="containerd://<SEQ>"
old template:  Container exited normally pod <*> podUID <UID> containerName nginx containerID containerd //<SEQ>
new template:  Container exited normally pod <*> podUID <UID> containerName <*> containerID containerd //<SEQ> 

content:  Waited for 459.0136ms due to client-side throttling, not priority and fairness, request: GET:https://kind-control-plane:6443/api/v1/namespaces/persistent-local-volumes-test-9176/persistentvolumeclaims/pvc-bn8k2
old template:  Waited for 662.255631ms due to client-side throttling not priority and fairness request GET https //kind-control-plane 6443/api/v1/namespaces/persistent-local-volumes-test-9176/persistentvolumeclaims/pvc-bn8k2
new template:  Waited for <*> due to client-side throttling not priority and fairness request GET https //kind-control-plane 6443/api/v1/namespaces/persistent-local-volumes-test-9176/persistentvolumeclaims/pvc-bn8k2 

content:  "Killing container with a grace period override" pod="ephemeral-1801-9055/csi-hostpath-snapshotter-0" podUID=<UID> containerName="csi-snapshotter" containerID="containerd://<SEQ>" gracePeriod=30
old template:  Killing container with a grace period override pod <*> podUID <UID> containerName <*> containerID containerd //<SEQ> gracePeriod 2
new template:  Killing container with a grace period override pod <*> podUID <UID> containerName <*> containerID containerd //<SEQ> gracePeriod <*> 

content:  "HTTP" verb="POST
old template:  HTTP verb GET
new template:  HTTP verb <*> 

content:  "Status for pod is up-to-date" pod="statefulset-5116/ss-0" statusVersion=3
old template:  Status for pod is up-to-date pod ephemeral-1801-9055/csi-hostpath-snapshotter-0 statusVersion 3
new template:  Status for pod is up-to-date pod <*> statusVersion 3 

content:  "Pod is terminated, but some containers are still running" pod="statefulset-5116/ss-0"
old template:  Pod is terminated but some containers are still running pod ephemeral-1801-9055/csi-hostpath-snapshotter-0
new template:  Pod is terminated but some containers are still running pod <*> 

content:  "Pod fully terminated and removed from etcd" pod="statefulset-5116/ss-0"
old template:  Pod fully terminated and removed from etcd pod ephemeral-1801-9055/csi-hostpath-snapshotter-0
new template:  Pod fully terminated and removed from etcd pod <*> 

content:  Waited for 750.403491ms due to client-side throttling, not priority and fairness, request: GET:https://kind-control-plane:6443/api/v1/namespaces/volume-4171/persistentvolumeclaims/csi-hostpathtvtkb
old template:  Waited for <*> due to client-side throttling not priority and fairness request GET https //kind-control-plane 6443/api/v1/namespaces/persistent-local-volumes-test-9176/persistentvolumeclaims/pvc-bn8k2
new template:  Waited for <*> due to client-side throttling not priority and fairness request GET https //kind-control-plane <*> 

content:  "Pod has completed execution and should be deleted from the API server" pod="volume-4171-3969/csi-hostpath-snapshotter-0" syncType="sync"
old template:  Pod has completed execution and should be deleted from the API server pod <*> syncType update
new template:  Pod has completed execution and should be deleted from the API server pod <*> syncType <*> 

content:  "Pod still has one or more containers in the non-exited state and will not be removed from desired state" pod="dns-6569/dns-test-<UID>"
old template:  Pod still has one or more containers in the non-exited state and will not be removed from desired state pod gc-1230/simpletest.rc-42h87
new template:  Pod still has one or more containers in the non-exited state and will not be removed from desired state pod <*> 

content:  "computePodActions got for pod" podActions={KillPod:true CreateSandbox:false SandboxID:<SEQ> Attempt:1 NextInitContainerToStart:nil ContainersToStart:[] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="downward-api-1800/downwardapi-volume-<UID>"
old template:  computePodActions got for pod podActions KillPod true CreateSandbox false SandboxID <SEQ> Attempt 0 NextInitContainerToStart nil ContainersToStart ContainersToKill map EphemeralContainersToStart pod downward-api-1800/downwardapi-volume-<UID>
new template:  computePodActions got for pod podActions KillPod true CreateSandbox false SandboxID <SEQ> Attempt <*> NextInitContainerToStart nil ContainersToStart ContainersToKill map EphemeralContainersToStart pod downward-api-1800/downwardapi-volume-<UID> 

content:  "Status for pod is up-to-date" pod="volume-expand-1966-4028/csi-hostpath-provisioner-0" statusVersion=4
old template:  Status for pod is up-to-date pod <*> statusVersion 3
new template:  Status for pod is up-to-date pod <*> statusVersion <*> 

content:  "Probe succeeded" probeType="Readiness" pod="nettest-137/netserver-0" podUID=<UID> containerName="webserver"
old template:  Probe succeeded probeType Liveness pod nettest-137/netserver-0 podUID <UID> containerName webserver
new template:  Probe succeeded probeType <*> pod nettest-137/netserver-0 podUID <UID> containerName webserver 

content:  "Already ran container, do nothing" pod="provisioning-8269/pod-subpath-test-preprovisionedpv-hwlp" containerName="test-container-subpath-preprovisionedpv-hwlp"
old template:  Already ran container do nothing pod downward-api-1800/downwardapi-volume-<UID> containerName client-container
new template:  Already ran container do nothing pod <*> containerName <*> 

content:  "Container of pod is not in the desired state and shall be started" containerName="test-container-volume-inlinevolume-tw7q" pod="provisioning-9207/pod-subpath-test-inlinevolume-tw7q"
old template:  Container of pod is not in the desired state and shall be started containerName test-container-subpath-inlinevolume-tw7q pod provisioning-9207/pod-subpath-test-inlinevolume-tw7q
new template:  Container of pod is not in the desired state and shall be started containerName <*> pod provisioning-9207/pod-subpath-test-inlinevolume-tw7q 

content:  "Pod has completed and its containers have been terminated, ignoring remaining sync work" pod="provisioning-9207/pod-subpath-test-inlinevolume-tw7q" syncType="sync"
old template:  Pod has completed and its containers have been terminated ignoring remaining sync work pod downward-api-1800/downwardapi-volume-<UID> syncType sync
new template:  Pod has completed and its containers have been terminated ignoring remaining sync work pod <*> syncType sync 

content:  "computePodActions got for pod" podActions={KillPod:true CreateSandbox:false SandboxID:<SEQ> Attempt:0 NextInitContainerToStart:nil ContainersToStart:[] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="provisioning-9207/pod-subpath-test-inlinevolume-tw7q"
old template:  computePodActions got for pod podActions KillPod true CreateSandbox false SandboxID <SEQ> Attempt <*> NextInitContainerToStart nil ContainersToStart ContainersToKill map EphemeralContainersToStart pod downward-api-1800/downwardapi-volume-<UID>
new template:  computePodActions got for pod podActions KillPod true CreateSandbox false SandboxID <SEQ> Attempt <*> NextInitContainerToStart nil ContainersToStart ContainersToKill map EphemeralContainersToStart pod <*> 

content:  "Stopping PodSandbox for pod, because all other containers are dead" pod="provisioning-9207/pod-subpath-test-inlinevolume-tw7q"
old template:  Stopping PodSandbox for pod because all other containers are dead pod downward-api-1800/downwardapi-volume-<UID>
new template:  Stopping PodSandbox for pod because all other containers are dead pod <*> 

content:  Received secret secrets-2296/s-test-opt-del-<UID> containing (0) pieces of data, 0 total bytes
old template:  Received secret secrets-2296/s-test-opt-del-<UID> containing 1 pieces of data 7 total bytes
new template:  Received secret secrets-2296/s-test-opt-del-<UID> containing <*> pieces of data <*> total bytes 

content:  "Completed init container for pod" containerName="init-volume-preprovisionedpv-wgm2" pod="provisioning-3348/pod-subpath-test-preprovisionedpv-wgm2"
old template:  Completed init container for pod containerName test-init-subpath-inlinevolume-tw7q pod provisioning-9207/pod-subpath-test-inlinevolume-tw7q
new template:  Completed init container for pod containerName <*> pod <*> 

content:  "Container of pod is not in the desired state and shall be started" containerName="test-container-subpath-preprovisionedpv-wgm2" pod="provisioning-3348/pod-subpath-test-preprovisionedpv-wgm2"
old template:  Container of pod is not in the desired state and shall be started containerName <*> pod provisioning-9207/pod-subpath-test-inlinevolume-tw7q
new template:  Container of pod is not in the desired state and shall be started containerName <*> pod <*> 

content:  "No ready sandbox for pod can be found. Need to start a new one" pod="provisioning-3348/pod-subpath-test-preprovisionedpv-wgm2"
old template:  No ready sandbox for pod can be found. Need to start a new one pod downward-api-1800/downwardapi-volume-<UID>
new template:  No ready sandbox for pod can be found. Need to start a new one pod <*> 

content:  "Creating hosts mount for container" pod="provisioning-8378/hostexec-kind-worker-bsglp" containerName="agnhost-container" podIPs=[<ID>::2] path=true
old template:  Creating hosts mount for container pod provisioning-3348/hostexec-kind-worker-5dlsr containerName agnhost-container podIPs <ID> 2 path true
new template:  Creating hosts mount for container pod <*> containerName agnhost-container podIPs <ID> 2 path true 

content:  Waited for 186.232957ms due to client-side throttling, not priority and fairness, request: PATCH:https://kind-control-plane:6443/api/v1/namespaces/provisioning-1607/pods/hostpath-symlink-prep-provisioning-1607/status
old template:  Waited for <*> due to client-side throttling not priority and fairness request GET https //kind-control-plane <*>
new template:  Waited for <*> due to client-side throttling not priority and fairness request <*> https //kind-control-plane <*> 

content:  "No status for pod" pod="statefulset-5071/ss2-1"
old template:  No status for pod pod nettest-137/netserver-0
new template:  No status for pod pod <*> 

content:  "Container readiness unchanged" ready=false pod="statefulset-5071/ss2-1" containerID="containerd://<SEQ>"
old template:  Container readiness unchanged ready false pod nettest-137/netserver-0 containerID containerd //<SEQ>
new template:  Container readiness unchanged ready false pod <*> containerID containerd //<SEQ> 

content:  "Probe succeeded" probeType="Readiness" pod="statefulset-5071/ss2-1" podUID=<UID> containerName="webserver"
old template:  Probe succeeded probeType <*> pod nettest-137/netserver-0 podUID <UID> containerName webserver
new template:  Probe succeeded probeType <*> pod <*> podUID <UID> containerName webserver 

content:  "Probe failed" probeType="Readiness" pod="statefulset-5071/ss2-1" podUID=<UID> containerName="webserver" probeResult=failure output="Get \"http://[fd00:10:244:1::196]:80/index.html\": context deadline exceeded (Client.Timeout exceeded while awaiting headers)"
old template:  Probe failed probeType Readiness pod statefulset-5071/ss2-1 podUID <UID> containerName webserver probeResult failure output Get http // fd00 10 244 1 17e 80/index.html context deadline exceeded Client.Timeout exceeded while awaiting headers
new template:  Probe failed probeType Readiness pod statefulset-5071/ss2-1 podUID <UID> containerName webserver probeResult failure output Get http // fd00 10 244 1 <*> 80/index.html context deadline exceeded Client.Timeout exceeded while awaiting headers 

Processing line: 200, rate 2100.8 lines/sec, 73 clusters so far.
content:  "Pod is terminated, but some containers have not been cleaned up" pod="statefulset-5071/ss2-1
old template:  Pod is terminated but some volumes have not been cleaned up pod volume-expand-1966-4028/csi-hostpath-provisioner-0
new template:  Pod is terminated but some <*> have not been cleaned up pod <*> 

content:  "computePodActions got for pod" podActions={KillPod:true CreateSandbox:true SandboxID: Attempt:0 NextInitContainerToStart:nil ContainersToStart:[0 1 2] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="dns-3927/dns-test-<UID>"
old template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID Attempt 0 NextInitContainerToStart nil ContainersToStart 0 1 2 ContainersToKill map EphemeralContainersToStart pod dns-6569/dns-test-<UID>
new template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID Attempt 0 NextInitContainerToStart nil ContainersToStart 0 1 2 ContainersToKill map EphemeralContainersToStart pod <*> 

content:  "Pod is terminated, but pod cgroup sandbox has not been cleaned up" pod="csi-mock-volumes-6281-2412/csi-mockplugin-attacher-0"
old template:  Pod is terminated but pod cgroup sandbox has not been cleaned up pod ephemeral-9870-4082/csi-hostpath-snapshotter-0
new template:  Pod is terminated but pod cgroup sandbox has not been cleaned up pod <*> 

content:  "computePodActions got for pod" podActions={KillPod:true CreateSandbox:true SandboxID: Attempt:0 NextInitContainerToStart:nil ContainersToStart:[0 1] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="replicaset-2941/test-rs-q7js9"
old template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID Attempt 0 NextInitContainerToStart nil ContainersToStart 0 1 ContainersToKill map EphemeralContainersToStart pod provisioning-8269/pod-subpath-test-preprovisionedpv-hwlp
new template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID Attempt 0 NextInitContainerToStart nil ContainersToStart 0 1 ContainersToKill map EphemeralContainersToStart pod <*> 

content:  "Probe failed" probeType="Readiness" pod="container-probe-5887/test-webserver-<UID>" podUID=<UID> containerName="test-webserver" probeResult=failure output="Get \"http://[fd00:10:244:1::b5]:81/\": context deadline exceeded (Client.Timeout exceeded while awaiting headers)"
old template:  Probe failed probeType Readiness pod statefulset-5071/ss2-1 podUID <UID> containerName webserver probeResult failure output Get http // fd00 10 244 1 <*> 80/index.html context deadline exceeded Client.Timeout exceeded while awaiting headers
new template:  Probe failed probeType Readiness pod <*> podUID <UID> containerName <*> probeResult failure output Get http // fd00 10 244 1 <*> <*> context deadline exceeded Client.Timeout exceeded while awaiting headers 

content:  "Pod status is inconsistent with cached status for pod, a reconciliation should be triggered" pod="kubelet-2846/cleanup20-<UID>-pngf6
old template:  Pod status is inconsistent with cached status for pod a reconciliation should be triggered pod provisioning-8378/hostexec-kind-worker-bsglp
new template:  Pod status is inconsistent with cached status for pod a reconciliation should be triggered pod <*> 

content:  "Unable to attach or mount volumes for pod; skipping pod" err="unmounted volumes=[kube-api-access-prws8], unattached volumes=[kube-api-access-prws8]: timed out waiting for the condition" pod="apply-4566/deployment-shared-unset-55bfccbb6c-vgcb9"
old template:  Unable to attach or mount volumes for pod; skipping pod err unmounted volumes kube-api-access-nlbt2 unattached volumes kube-api-access-nlbt2 timed out waiting for the condition pod pods-3446/pod-submit-status-0-11
new template:  Unable to attach or mount volumes for pod; skipping pod err unmounted volumes <*> unattached volumes <*> timed out waiting for the condition pod <*> 

content:  "Error syncing pod, skipping" err="unmounted volumes=[kube-api-access-prws8], unattached volumes=[kube-api-access-prws8]: timed out waiting for the condition
old template:  Error syncing pod skipping err unmounted volumes kube-api-access-nlbt2 unattached volumes kube-api-access-nlbt2 timed out waiting for the condition
new template:  Error syncing pod skipping err unmounted volumes <*> unattached volumes <*> timed out waiting for the condition 

content:  "Probe target container not found" pod="volumemode-9000-9501/csi-hostpathplugin-0" containerName="hostpath"
old template:  Probe target container not found pod statefulset-5071/ss2-1 containerName webserver
new template:  Probe target container not found pod <*> containerName <*> 

content:  "Probe succeeded" probeType="Liveness" pod="volumemode-9000-9501/csi-hostpathplugin-0" podUID=<UID> containerName="hostpath"
old template:  Probe succeeded probeType <*> pod <*> podUID <UID> containerName webserver
new template:  Probe succeeded probeType <*> pod <*> podUID <UID> containerName <*> 

content:  "Probe failed" probeType="Liveness" pod="volumemode-9000-9501/csi-hostpathplugin-0" podUID=<UID> containerName="hostpath" probeResult=failure output="Get \"http://[fd00:10:244:1::80]:9898/healthz\": dial tcp [fd00:10:244:1::80]:9898: connect: connection refused"
old template:  Probe failed probeType Readiness pod container-probe-5887/test-webserver-<UID> podUID <UID> containerName test-webserver probeResult failure output Get http // fd00 10 244 1 b5 81/ dial tcp fd00 10 244 1 b5 81 connect connection refused
new template:  Probe failed probeType <*> pod <*> podUID <UID> containerName <*> probeResult failure output Get http // fd00 10 244 1 <*> <*> dial tcp fd00 10 244 1 <*> <*> connect connection refused 

content:  "Non-running container probed" pod="volumemode-9000-9501/csi-hostpathplugin-0" containerName="hostpath"
old template:  Non-running container probed pod statefulset-5071/ss2-1 containerName webserver
new template:  Non-running container probed pod <*> containerName <*> 

content:  "SyncLoop (probe)" probe="liveness
old template:  SyncLoop probe probe readiness
new template:  SyncLoop probe probe <*> 

content:  "Failed to create sandbox for pod" err="rpc error: code = DeadlineExceeded desc = context deadline exceeded" pod="ephemeral-6817/inline-volume-tester-8fxzp"
old template:  Failed to create sandbox for pod err rpc error code DeadlineExceeded desc context deadline exceeded pod kubelet-2846/cleanup20-<UID>-pngf6
new template:  Failed to create sandbox for pod err rpc error code DeadlineExceeded desc context deadline exceeded pod <*> 

content:  "CreatePodSandbox for pod failed" err="rpc error: code = DeadlineExceeded desc = context deadline exceeded" pod="ephemeral-6817/inline-volume-tester-8fxzp"
old template:  CreatePodSandbox for pod failed err rpc error code DeadlineExceeded desc context deadline exceeded pod kubelet-2846/cleanup20-<UID>-pngf6
new template:  CreatePodSandbox for pod failed err rpc error code DeadlineExceeded desc context deadline exceeded pod <*> 

content:  "computePodActions got for pod" podActions={KillPod:false CreateSandbox:false SandboxID:<SEQ> Attempt:0 NextInitContainerToStart:nil ContainersToStart:[0 1] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="provisioning-242/pod-subpath-test-preprovisionedpv-xwvx"
old template:  computePodActions got for pod podActions KillPod false CreateSandbox false SandboxID <SEQ> Attempt 0 NextInitContainerToStart nil ContainersToStart 0 1 ContainersToKill map EphemeralContainersToStart pod provisioning-9207/pod-subpath-test-inlinevolume-tw7q
new template:  computePodActions got for pod podActions KillPod false CreateSandbox false SandboxID <SEQ> Attempt 0 NextInitContainerToStart nil ContainersToStart 0 1 ContainersToKill map EphemeralContainersToStart pod <*> 

Processing line: 200, rate 4781.5 lines/sec, 92 clusters so far.
content:  "Pod was deleted and then recreated, skipping status update" pod="statefulset-5071/ss2-0" oldPodUID=<UID> podUID=<UID>
old template:  Pod was deleted and then recreated skipping status update pod statefulset-5116/ss-0 oldPodUID <UID> podUID <UID>
new template:  Pod was deleted and then recreated skipping status update pod <*> oldPodUID <UID> podUID <UID> 

Processing line: 200, rate 4000.4 lines/sec, 99 clusters so far.
content:  "computePodActions got for pod" podActions={KillPod:false CreateSandbox:false SandboxID:<SEQ> Attempt:0 NextInitContainerToStart:nil ContainersToStart:[0] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="provisioning-9293/pod-subpath-test-dynamicpv-cdm6"
old template:  computePodActions got for pod podActions KillPod false CreateSandbox false SandboxID <SEQ> Attempt 0 NextInitContainerToStart nil ContainersToStart 0 ContainersToKill map EphemeralContainersToStart pod provisioning-3348/pod-subpath-test-preprovisionedpv-wgm2
new template:  computePodActions got for pod podActions KillPod false CreateSandbox false SandboxID <SEQ> Attempt 0 NextInitContainerToStart nil ContainersToStart 0 ContainersToKill map EphemeralContainersToStart pod <*> 

content:  kubernetes.io/csi: calling NodeUnpublishVolume rpc: [volid=csi-<SEQ>, target_path=/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
old template:  kubernetes.io/csi calling NodePublishVolume rpc volid csi-<SEQ> target_path /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
new template:  kubernetes.io/csi calling <*> rpc volid csi-<SEQ> target_path /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount 

content:  Couldn't get secret pv-4669/secret-pod-ephm-test: secret "secret-pod-ephm-test" not found
old template:  Couldn't get configMap apply-4900/kube-root-ca.crt configmap kube-root-ca.crt not found
new template:  Couldn't get <*> <*> <*> <*> not found 

content:  Received secret subpath-382/my-secret containing (1) pieces of data, 12 total bytes
old template:  Received secret secrets-2296/s-test-opt-del-<UID> containing <*> pieces of data <*> total bytes
new template:  Received secret <*> containing <*> pieces of data <*> total bytes 

content:  "getSandboxIDByPodUID got sandbox IDs for pod" podSandboxID=[] pod="services-2051/pod2"
old template:  getSandboxIDByPodUID got sandbox IDs for pod podSandboxID pod gc-1230/simpletest.rc-8t889
new template:  getSandboxIDByPodUID got sandbox IDs for pod podSandboxID pod <*> 

content:  Couldn't get configMap replicaset-3311/kube-root-ca.crt: object "replicaset-3311"/"kube-root-ca.crt" not registered
old template:  Couldn't get configMap deployment-7048/kube-root-ca.crt object deployment-7048 / kube-root-ca.crt not registered
new template:  Couldn't get configMap <*> object <*> / kube-root-ca.crt not registered 

content:  Error preparing data for projected volume kube-api-access-fvppn for pod webhook-8229/webhook-to-be-mutated: failed to fetch token: serviceaccounts "default" is forbidden: unable to create new content in namespace webhook-8229 because it is being terminated
old template:  Error preparing data for projected volume kube-api-access-grbfw for pod deployment-7048/webserver-deployment-847dcfb7fb-jvpvm failed to fetch token serviceaccounts default is forbidden unable to create new content in namespace deployment-7048 because it is being terminated
new template:  Error preparing data for projected volume <*> for pod <*> failed to fetch token serviceaccounts default is forbidden unable to create new content in namespace <*> because it is being terminated 

content:  "Actual state does not yet have volume mount information and pod still exists in pod manager, skip removing volume from desired state" pod="webhook-8229/webhook-to-be-mutated" podUID=<UID> volumeName="kube-api-access-fvppn"
old template:  Actual state does not yet have volume mount information and pod still exists in pod manager skip removing volume from desired state pod pods-3446/pod-submit-status-1-6 podUID <UID> volumeName kube-api-access-wz5nb
new template:  Actual state does not yet have volume mount information and pod still exists in pod manager skip removing volume from desired state pod <*> podUID <UID> volumeName <*> 

content:  "Skipping unused volume" pod="volumemode-903/pod-<UID>" volumeName="volume1"
old template:  Skipping unused volume pod volumemode-9000/pod-<UID> volumeName volume1
new template:  Skipping unused volume pod <*> volumeName volume1 

content:  Bound SubPath /tmp/provisioning-8176/provisioning-8176 into /var/lib/kubelet/pods/<UID>/volume-subpaths/test-volume/test-container-subpath-inlinevolume-t6nr/0
old template:  Bound SubPath /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~empty-dir/test-volume/provisioning-2179 into /var/lib/kubelet/pods/<UID>/volume-subpaths/test-volume/test-container-subpath-inlinevolume-7925/0
new template:  Bound SubPath <*> into <*> 

content:  Error preparing data for projected volume kube-api-access-vg9px for pod deployment-7048/webserver-deployment-847dcfb7fb-2nw9q: [failed to fetch token: serviceaccounts "default" is forbidden: User "system:node:kind-worker" cannot create resource "serviceaccounts/token" in API group "" in the namespace "deployment-7048": no relationship found between node 'kind-worker' and this object, object "deployment-7048"/"kube-root-ca.crt" not registered]
old template:  Error preparing data for projected volume kube-api-access-grbfw for pod deployment-7048/webserver-deployment-847dcfb7fb-jvpvm failed to fetch token serviceaccounts default is forbidden User system node kind-worker cannot create resource serviceaccounts/token in API group in the namespace deployment-7048 no relationship found between node 'kind-worker' and this object object deployment-7048 / kube-root-ca.crt not registered
new template:  Error preparing data for projected volume <*> for pod <*> failed to fetch token serviceaccounts default is forbidden User system node kind-worker cannot create resource serviceaccounts/token in API group in the namespace deployment-7048 no relationship found between node 'kind-worker' and this object object deployment-7048 / kube-root-ca.crt not registered 

content:  Error preparing data for projected volume kube-api-access-nhlx6 for pod deployment-4859/webserver-847dcfb7fb-fkrz9: failed to fetch token: pod "webserver-847dcfb7fb-fkrz9" not found
old template:  Error preparing data for projected volume kube-api-access-88nkr for pod deployment-4859/webserver-847dcfb7fb-gmbsn failed to fetch token pod webserver-847dcfb7fb-gmbsn not found
new template:  Error preparing data for projected volume <*> for pod <*> failed to fetch token pod <*> not found 

content:  Couldn't get configMap configmap-3691/configmap-test-volume-<UID>: failed to sync configmap cache: timed out waiting for the condition
old template:  Couldn't get secret secrets-2296/s-test-opt-del-<UID> failed to sync secret cache timed out waiting for the condition
new template:  Couldn't get <*> <*> failed to sync <*> cache timed out waiting for the condition 

content:  Received configMap configmap-3691/configmap-test-volume-<UID> containing (3) pieces of data, 21 total bytes
old template:  Received configMap configmap-9309/configmap-test-volume-map-<UID> containing 3 pieces of data 21 total bytes
new template:  Received configMap <*> containing 3 pieces of data 21 total bytes 

content:  "Container startup unchanged" pod="container-probe-9287/startup-<UID>" containerID="containerd://<SEQ>"
old template:  Container startup unchanged pod container-probe-9864/startup-<UID> containerID containerd //<SEQ>
new template:  Container startup unchanged pod <*> containerID containerd //<SEQ> 

content:  "Exec-Probe runProbe" pod="container-probe-9287/startup-<UID>" containerName="busybox" execCommand=[/bin/false]
old template:  Exec-Probe runProbe pod container-probe-9864/startup-<UID> containerName busybox execCommand /bin/false
new template:  Exec-Probe runProbe pod <*> containerName busybox execCommand /bin/false 

content:  "Probe failed" probeType="Startup" pod="container-probe-9287/startup-<UID>" podUID=<UID> containerName="busybox" probeResult=failure output=""
old template:  Probe failed probeType Liveness pod container-probe-9864/startup-<UID> podUID <UID> containerName busybox probeResult failure output
new template:  Probe failed probeType <*> pod <*> podUID <UID> containerName busybox probeResult failure output 

Processing line: 200, rate 2323.2 lines/sec, 123 clusters so far.
content:  "Probe failed" probeType="Readiness" pod="statefulset-2741/ss2-1" podUID=<UID> containerName="webserver" probeResult=failure output="Get \"http://[fd00:10:244:1::11e]:80/index.html\": dial tcp [fd00:10:244:1::11e]:80: i/o timeout (Client.Timeout exceeded while awaiting headers)"
old template:  Probe failed probeType Readiness pod statefulset-5071/ss2-1 podUID <UID> containerName webserver probeResult failure output Get http // fd00 10 244 1 17e 80/index.html dial tcp fd00 10 244 1 17e 80 i/o timeout Client.Timeout exceeded while awaiting headers
new template:  Probe failed probeType Readiness pod <*> podUID <UID> containerName webserver probeResult failure output Get http // fd00 10 244 1 <*> 80/index.html dial tcp fd00 10 244 1 <*> 80 i/o timeout Client.Timeout exceeded while awaiting headers 

content:  "Container readiness changed for unknown container" pod="statefulset-2741/ss2-1" containerID="containerd://<SEQ>"
old template:  Container readiness changed for unknown container pod statefulset-5955/ss2-2 containerID containerd //<SEQ>
new template:  Container readiness changed for unknown container pod <*> containerID containerd //<SEQ> 

Processing line: 200, rate 2996.4 lines/sec, 123 clusters so far.
content:  "Checking backoff for container in pod" containerName="agnhost-container" pod="services-2562/kube-proxy-mode-detector"
old template:  Checking backoff for container in pod containerName csi-provisioner pod csi-mock-volumes-9390-6343/csi-mockplugin-0
new template:  Checking backoff for container in pod containerName <*> pod <*> 

content:  Error preparing data for projected volume kube-api-access-f9x5v for pod disruption-6959/rs-z44jz: [failed to fetch token: serviceaccounts "default" is forbidden: unable to create new content in namespace disruption-6959 because it is being terminated, configmap "kube-root-ca.crt" not found]
old template:  Error preparing data for projected volume kube-api-access-fvppn for pod webhook-8229/webhook-to-be-mutated failed to fetch token serviceaccounts default is forbidden unable to create new content in namespace webhook-8229 because it is being terminated configmap kube-root-ca.crt not found
new template:  Error preparing data for projected volume <*> for pod <*> failed to fetch token serviceaccounts default is forbidden unable to create new content in namespace <*> because it is being terminated configmap kube-root-ca.crt not found 

Processing line: 200, rate 4810.2 lines/sec, 124 clusters so far.
Processing line: 200, rate 3897.3 lines/sec, 128 clusters so far.
content:  "Already successfully ran container, do nothing" pod="pods-7543/pod-always-succeed92165fd8-078f-4444-aad7-1e5a26e0a4dd" containerName="bar"
old template:  Already successfully ran container do nothing pod kubectl-7863/run-log-test containerName run-log-test
new template:  Already successfully ran container do nothing pod <*> containerName <*> 

content:  kubernetes.io/csi: saving volume data file [/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/vol_data.json]
old template:  kubernetes.io/csi saving volume data file /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/vol_data.json
new template:  kubernetes.io/csi saving volume data file <*> 

content:  kubernetes.io/csi: volume data file saved successfully [/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/vol_data.json]
old template:  kubernetes.io/csi volume data file saved successfully /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/vol_data.json
new template:  kubernetes.io/csi volume data file saved successfully <*> 

content:  kubernetes.io/csi: loading volume data file [/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/vol_data.json]
old template:  kubernetes.io/csi loading volume data file /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/vol_data.json
new template:  kubernetes.io/csi loading volume data file <*> 

content:  kubernetes.io/csi: also deleting volume info data file [/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/vol_data.json]
old template:  kubernetes.io/csi also deleting volume info data file /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/vol_data.json
new template:  kubernetes.io/csi also deleting volume info data file <*> 

content:  Error preparing data for projected volume kube-api-access-5zcg8 for pod persistent-local-volumes-test-4047/pod-<UID>: [failed to fetch token: serviceaccounts "default" is forbidden: User "system:node:kind-worker" cannot create resource "serviceaccounts/token" in API group "" in the namespace "persistent-local-volumes-test-4047": no relationship found between node 'kind-worker' and this object, object "persistent-local-volumes-test-4047"/"kube-root-ca.crt" not registered]
old template:  Error preparing data for projected volume <*> for pod <*> failed to fetch token serviceaccounts default is forbidden User system node kind-worker cannot create resource serviceaccounts/token in API group in the namespace deployment-7048 no relationship found between node 'kind-worker' and this object object deployment-7048 / kube-root-ca.crt not registered
new template:  Error preparing data for projected volume <*> for pod <*> failed to fetch token serviceaccounts default is forbidden User system node kind-worker cannot create resource serviceaccounts/token in API group in the namespace <*> no relationship found between node 'kind-worker' and this object object <*> / kube-root-ca.crt not registered 

content:  pod configmap-9019/pod-configmaps-<UID> volume createcm-volume: write required for target directory /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~configmap/createcm-volume
old template:  pod configmap-9019/pod-configmaps-<UID> volume deletecm-volume write required for target directory /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~configmap/deletecm-volume
new template:  pod configmap-9019/pod-configmaps-<UID> volume <*> write required for target directory <*> 

content:  Setting up a downwardAPI volume podinfo for pod downward-api-1757/annotationupdate374fa10c-78aa-409a-9cd3-dbfa326984fb at /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~downward-api/podinfo
old template:  Setting up a downwardAPI volume podinfo for pod downward-api-1800/downwardapi-volume-<UID> at /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~downward-api/podinfo
new template:  Setting up a downwardAPI volume podinfo for pod <*> at /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~downward-api/podinfo 

content:  pod downward-api-1757/annotationupdate374fa10c-78aa-409a-9cd3-dbfa326984fb volume podinfo: write required for target directory /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~downward-api/podinfo
old template:  pod configmap-9019/pod-configmaps-<UID> volume <*> write required for target directory <*>
new template:  pod <*> volume <*> write required for target directory <*> 

Processing line: 200, rate 5028.4 lines/sec, 131 clusters so far.
content:  "Probe failed" probeType="Readiness" pod="nettest-8956/netserver-0" podUID=<UID> containerName="webserver" probeResult=failure output="Get \"http://[fd00:10:244:1::117]:8080/healthz\": dial tcp [fd00:10:244:1::117]:8080: connect: network is unreachable"
old template:  Probe failed probeType Liveness pod provisioning-8683-8677/csi-hostpathplugin-0 podUID <UID> containerName hostpath probeResult failure output Get http // fd00 10 244 1 1f3 9898/healthz dial tcp fd00 10 244 1 1f3 9898 connect network is unreachable
new template:  Probe failed probeType <*> pod <*> podUID <UID> containerName <*> probeResult failure output Get http // fd00 10 244 1 <*> <*> dial tcp fd00 10 244 1 <*> <*> connect network is unreachable 

content:  "Creating hosts mount for container" pod="downward-api-6236/downward-api-<UID>" containerName="dapi-container" podIPs=[<ID>::2] path=true
old template:  Creating hosts mount for container pod <*> containerName agnhost-container podIPs <ID> 2 path true
new template:  Creating hosts mount for container pod <*> containerName <*> podIPs <ID> 2 path true 

content:  Received configMap configmap-9019/cm-test-opt-create-<UID> containing (0) pieces of data, 0 total bytes
old template:  Received configMap <*> containing 3 pieces of data 21 total bytes
new template:  Received configMap <*> containing <*> pieces of data <*> total bytes 

content:  "computePodActions got for pod" podActions={KillPod:true CreateSandbox:true SandboxID: Attempt:0 NextInitContainerToStart:nil ContainersToStart:[0 1 2 3] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="csi-mock-volumes-3084-6830/csi-mockplugin-0"
old template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID Attempt 0 NextInitContainerToStart nil ContainersToStart 0 1 2 3 ContainersToKill map EphemeralContainersToStart pod csi-mock-volumes-9390-6343/csi-mockplugin-0
new template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID Attempt 0 NextInitContainerToStart nil ContainersToStart 0 1 2 3 ContainersToKill map EphemeralContainersToStart pod <*> 

Processing line: 200, rate 3662.3 lines/sec, 133 clusters so far.
Processing line: 200, rate 3548.7 lines/sec, 134 clusters so far.
Processing line: 400, rate 5158.2 lines/sec, 134 clusters so far.
Processing line: 600, rate 5362.8 lines/sec, 134 clusters so far.
content:  "Failed to delete status for pod" pod="ephemeral-1801/inline-volume-tester-n99kv" err="pods \"inline-volume-tester-n99kv\" not found"
old template:  Failed to delete status for pod pod persistent-local-volumes-test-2836/pod-<UID> err pods pod-<UID> not found
new template:  Failed to delete status for pod pod <*> err pods <*> not found 

content:  "Exec-Probe runProbe" pod="container-probe-3551/busybox-<UID>" containerName="busybox" execCommand=[cat /tmp/health]
old template:  Exec-Probe runProbe pod container-probe-9864/startup-<UID> containerName busybox execCommand cat /tmp/startup
new template:  Exec-Probe runProbe pod <*> containerName busybox execCommand cat <*> 

content:  "Running preStop hook" pod="prestop-7504/tester" podUID=<UID> containerName="tester" containerID="containerd://<SEQ>"
old template:  Running preStop hook pod container-lifecycle-hook-7916/pod-with-prestop-http-hook podUID <UID> containerName pod-with-prestop-http-hook containerID containerd //<SEQ>
new template:  Running preStop hook pod <*> podUID <UID> containerName <*> containerID containerd //<SEQ> 

content:  "PreStop hook completed" pod="prestop-7504/tester" podUID=<UID> containerName="tester" containerID="containerd://<SEQ>"
old template:  PreStop hook completed pod container-lifecycle-hook-7916/pod-with-prestop-http-hook podUID <UID> containerName pod-with-prestop-http-hook containerID containerd //<SEQ>
new template:  PreStop hook completed pod <*> podUID <UID> containerName <*> containerID containerd //<SEQ> 

content:  "Probe failed" probeType="Readiness" pod="statefulset-8107/ss-1" podUID=<UID> containerName="webserver" probeResult=failure output=""
old template:  Probe failed probeType <*> pod <*> podUID <UID> containerName busybox probeResult failure output
new template:  Probe failed probeType <*> pod <*> podUID <UID> containerName <*> probeResult failure output 

Processing line: 200, rate 2344.2 lines/sec, 136 clusters so far.
Processing line: 200, rate 5028.8 lines/sec, 136 clusters so far.
Processing line: 400, rate 5211.2 lines/sec, 136 clusters so far.
content:  "Error processing volume" err="error processing PVC pvc-protection-735/pvc-protection54r7p: failed to fetch PVC from API server: persistentvolumeclaims \"pvc-protection54r7p\" is forbidden: User \"system:node:kind-worker\" cannot get resource \"persistentvolumeclaims\" in API group \"\" in the namespace \"pvc-protection-735\": no relationship found between node 'kind-worker' and this object" pod="pvc-protection-735/pvc-tester-bjv9z" volumeName="volume1"
old template:  Error processing volume err error processing PVC persistent-local-volumes-test-4047/pvc-hqgkd failed to fetch PVC from API server persistentvolumeclaims pvc-hqgkd is forbidden User system node kind-worker cannot get resource persistentvolumeclaims in API group in the namespace persistent-local-volumes-test-4047 no relationship found between node 'kind-worker' and this object pod persistent-local-volumes-test-4047/pod-<UID> volumeName volume1
new template:  Error processing volume err error processing PVC <*> failed to fetch PVC from API server persistentvolumeclaims <*> is forbidden User system node kind-worker cannot get resource persistentvolumeclaims in API group in the namespace <*> no relationship found between node 'kind-worker' and this object pod <*> volumeName volume1 

content:  kubernetes.io/csi: mounter.GetPath generated [/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/mount]
old template:  kubernetes.io/csi mounter.GetPath generated /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
new template:  kubernetes.io/csi mounter.GetPath generated <*> 

content:  kubernetes.io/csi: Mounter.SetUpAt(/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/mount)
old template:  kubernetes.io/csi Mounter.SetUpAt /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
new template:  kubernetes.io/csi Mounter.SetUpAt <*> 

content:  kubernetes.io/csi: calling NodePublishVolume rpc [volid=csi-<SEQ>,target_path=/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/mount]
old template:  kubernetes.io/csi calling <*> rpc volid csi-<SEQ> target_path /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
new template:  kubernetes.io/csi calling <*> rpc volid csi-<SEQ> target_path <*> 

content:  kubernetes.io/csi: mounter.SetUp successfully requested NodePublish [/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/mount]
old template:  kubernetes.io/csi mounter.SetUp successfully requested NodePublish /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
new template:  kubernetes.io/csi mounter.SetUp successfully requested NodePublish <*> 

content:  kubernetes.io/csi: Unmounter.TearDown(/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/mount)
old template:  kubernetes.io/csi Unmounter.TearDown /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
new template:  kubernetes.io/csi Unmounter.TearDown <*> 

content:  kubernetes.io/csi: removing mount path [/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/mount]
old template:  kubernetes.io/csi removing mount path /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
new template:  kubernetes.io/csi removing mount path <*> 

content:  kubernetes.io/csi: dir not mounted, deleting it [/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/mount]
old template:  kubernetes.io/csi dir not mounted deleting it /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
new template:  kubernetes.io/csi dir not mounted deleting it <*> 

content:  kubernetes.io/csi: mounter.TearDownAt successfully unmounted dir [/var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-0/mount]
old template:  kubernetes.io/csi mounter.TearDownAt successfully unmounted dir /var/lib/kubelet/pods/<UID>/volumes/kubernetes.io~csi/my-volume-1/mount
new template:  kubernetes.io/csi mounter.TearDownAt successfully unmounted dir <*> 

content:  "Killing unwanted container for pod" containerName="e2e-test-httpd-pod" containerID={Type:containerd ID:<SEQ>} pod="kubectl-2456/e2e-test-httpd-pod"
old template:  Killing unwanted container for pod containerName busybox containerID Type containerd ID <SEQ> pod container-probe-9864/startup-<UID>
new template:  Killing unwanted container for pod containerName <*> containerID Type containerd ID <SEQ> pod <*> 

Processing line: 200, rate 5087.9 lines/sec, 139 clusters so far.
Processing line: 400, rate 3870.4 lines/sec, 140 clusters so far.
content:  "Probe errored" err="rpc error: code = Unknown desc = failed to exec in container: container is in CONTAINER_EXITED state" probeType="Readiness" pod="port-forwarding-8045/pfpod" podUID=<UID> containerName="readiness"
old template:  Probe errored err rpc error code Unknown desc failed to exec in container container is in CONTAINER_EXITED state probeType Readiness pod statefulset-8107/ss-1 podUID <UID> containerName webserver
new template:  Probe errored err rpc error code Unknown desc failed to exec in container container is in CONTAINER_EXITED state probeType Readiness pod <*> podUID <UID> containerName <*> 

Processing line: 200, rate 5032.7 lines/sec, 141 clusters so far.
content:  "Probe failed" probeType="Readiness" pod="statefulset-8515/ss-1" podUID=<UID> containerName="webserver" probeResult=failure output="HTTP probe failed with statuscode: 404"
old template:  Probe failed probeType Readiness pod statefulset-5955/ss2-1 podUID <UID> containerName webserver probeResult failure output HTTP probe failed with statuscode 404
new template:  Probe failed probeType Readiness pod <*> podUID <UID> containerName webserver probeResult failure output HTTP probe failed with statuscode 404 

Processing line: 200, rate 3638.1 lines/sec, 141 clusters so far.
Processing line: 400, rate 5190.6 lines/sec, 141 clusters so far.
content:  "Exec-Probe runProbe" pod="services-5323/slow-terminating-unready-pod-6st5n" containerName="slow-terminating-unready-pod" execCommand=[/bin/false]
old template:  Exec-Probe runProbe pod <*> containerName busybox execCommand /bin/false
new template:  Exec-Probe runProbe pod <*> containerName <*> execCommand /bin/false 

Processing line: 200, rate 4830.0 lines/sec, 143 clusters so far.
Processing line: 400, rate 4960.5 lines/sec, 143 clusters so far.
content:  "Unable to attach or mount volumes for pod; skipping pod" err="unmounted volumes=[test-volume kube-api-access-q7t8t], unattached volumes=[test-volume kube-api-access-q7t8t]: timed out waiting for the condition" pod="pv-4669/pod-ephm-test-projected-fs97"
old template:  Unable to attach or mount volumes for pod; skipping pod err unmounted volumes my-volume kube-api-access-mgp54 unattached volumes my-volume kube-api-access-mgp54 timed out waiting for the condition pod csi-mock-volumes-8509/inline-volume-ngc9m
new template:  Unable to attach or mount volumes for pod; skipping pod err unmounted volumes <*> <*> unattached volumes <*> <*> timed out waiting for the condition pod <*> 

content:  "Error syncing pod, skipping" err="unmounted volumes=[test-volume kube-api-access-q7t8t], unattached volumes=[test-volume kube-api-access-q7t8t]: timed out waiting for the condition
old template:  Error syncing pod skipping err unmounted volumes my-volume kube-api-access-mgp54 unattached volumes my-volume kube-api-access-mgp54 timed out waiting for the condition
new template:  Error syncing pod skipping err unmounted volumes <*> <*> unattached volumes <*> <*> timed out waiting for the condition 

content:  "computePodActions got for pod" podActions={KillPod:true CreateSandbox:true SandboxID:<SEQ> Attempt:1 NextInitContainerToStart:nil ContainersToStart:[0] ContainersToKill:map[] EphemeralContainersToStart:[]} pod="kubectl-1506/pause"
old template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID <SEQ> Attempt 1 NextInitContainerToStart nil ContainersToStart 0 ContainersToKill map EphemeralContainersToStart pod services-2562/kube-proxy-mode-detector
new template:  computePodActions got for pod podActions KillPod true CreateSandbox true SandboxID <SEQ> Attempt 1 NextInitContainerToStart nil ContainersToStart 0 ContainersToKill map EphemeralContainersToStart pod <*> 

