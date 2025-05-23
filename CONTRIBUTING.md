# Contributing Guide

Thanks for taking the time to consider contributing! We very much appreciate your time and effort. This document outlines the many ways you can contribute to our project as well as some helpful information.

### Project roadmap


* Ensure compatibility with Isaac ROS.

* Periodically check if a process is alive; if not, respawn it (this should be straightforward once arbitrary process killing and spawning is implemented).

* Add functionality so that if there is an unknown value in the output signature, the Torch queue is used; otherwise, the CUDA queue is used.

* During testing, verify that the shapes of inputs and outputs match the expected data shapes.

* Improve CUDA stream management by overlapping memory transfers to the GPU with inference and avoiding the default stream for TensorRT enqueue.

* Add a semantic segmentation head from DINOv2.

* Revert the default depth head to its previous implementation, as it appears to be slower than the one from DAV2.

* Use enums instead of magic strings as keys for dictionaries that transport data, to ensure uniformity across architectures.

* Evaluate whether resizing everything to 518 pixels makes sense.

* Modify buffer locks to support concurrent reads and a single write.

* Implement semaphore-based access control: each read decrements the semaphore by one, and each write must acquire the semaphore N times (where N is the number of active reading processes).

* Support dynamic process management: when a reading process is killed, the main process should acquire the semaphore to decrease its internal counter; when a new process is spawned, the semaphore should be released to increase the counter.

* Note that while semaphore-based management works well for CPU reads and writes, GPU data (where bandwidth is limited) will still require sequential device-to-device memory copies.

* Consider using semaphores if, instead of copying, pointers to TensorRT are passed.

* Use the technique “Create a Torch tensor with a device pointer” (from PyTorch Forums) to avoid copying memory.

* Reduce the number of tensor copies inside the engine.

* Add unit tests for the engine.

* Add an option to register an ONNX engine in the registry without a class definition (e.g., by internally converting it to TensorRT).

* Provide input to the foundation model via either a small queue or a buffer to ensure the newest data is used.

* Implement dynamic process management (starting and killing processes as needed) to optimize GPU memory usage as the number of models increases.

* Extend the buffer: if one position is locked, attempt to access the previous position (if it contains newer data) to prevent all heads from waiting on a "high-demand" frame.

* If always retrieving the newest data is desired, provide two or more slots from which data can be read; however, note that this increases complexity.

* Add a configuration option to the buffer that allows the user to always process images sequentially, even if this introduces delay.

* Reduce the number of data copies when interacting with CUDA by passing pointers to the output locations so that TensorRT’s results are stored directly.

* Add a script to automatically download weights for the models used and add them to the registry, preparing a default setup.

### A few guidelines
* All the models should use batch dimension, even if set to 1, however the input and output queues should operate without the batch dimension already

* The models should also process everything in float16

* opencv wants to have dimensions as last so (1080,1920,1) and not (1,1080,1920) P.S. by 'want' I mean it needs to have it this way or it will throw a tantrum including invalid free()

### Useful commands
* Check CUDA version

   `nvcc --version`
* Check OS version

    `cat /etc/os-release `
* Run interactive bash session in docker

    `docker exec -it MY_AWESOME_DOCKER_CONTAINER /bin/bash`
* Find if a package is installed

    `dpkg -l | grep nvinfer` OR `dpkg-query -W tensorrt`
* Install package from another index

    `python3 -m pip install --extra-index-url https://pypi.nvidia.com tensorrt_libs`
* Call nsys from within a docker container after relevant socket has been added as a volume

    `/nsys/bin/nsys`
* Call trtexec from inside a docker container

    `/usr/src/tensorrt/bin/trtexec`
* Export onnx model to tensorrt inside a docker container using float16 precision

    `/usr/src/tensorrt/bin/trtexec --onnx=PATH_TO_ONNX_MODEL.onnx --saveEngine=DESTINATION_PATH.trt --fp16`
* Check amount of shared memory usd in docker

    `df -h | grep shm`

### Encountered errors
* **If you get error related to SSH_AUTH_SOCK when building docker container try:**

    `eval $(ssh-agent)`
* **IF you get `ImportError: libnv*****.so cannot open shared object file: No such file or directory`:**

    First try to install said package through `apt install` if it does not work try:

    `export LD_LIBRARY_PATH="/usr/local/cuda/compat:$LD_LIBRARY_PATH"`

### License

Our project has our licensing terms, including rules governing redistribution, documented in our [LICENSE](LICENSE) file. Please take a look at that file and ensure you understand the terms. This will impact how we, or others, use your contributions.


