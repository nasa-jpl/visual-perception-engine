# Visual Perception Engine
We created Visual Percpetion Engine to have one coherent module that will process visual percepts on GPU and output desired insights such as monocular depth. It was designed to be **efficient, modular, use constant memory and dynamic** such that it can be controlled at runtime. We provide an implementation that in **real time (30Hz)** computes monocular depth estimation (based on DepthAnythingV2), semantic segmentation and object detections. Furhtermore, the engine was created to be **easily configurable and extendable**, such that one can add their own model heads into the framework or change the existing ones. 

> [!TIP]
> We also provide C++ ROS2 (Humble) node that can be easily incorporated into existing robotic stacks.


## How does it work?
The engine levarages a single powerful vision foundation model (DINOv2 in our case) that computes features and several model heads that use those features to compute desired insights. Whenever possible we converted the models into TensorRT engine for efficiency, however our architecture also supports PyTorch models. Each model (foundation model and individual model heads) are run in separate processes, and send data between each other using custom queues and buffers that allow for sending pointers to tensors on GPU. See figure below for an overview:

![Figure 1: Overview of the engine's internal structure](static/perception_engine.png)

### Data transfer
Between all elements of the engine data should be exchanged in a dictionary format using meaningful keys for indetifiability. For example, DINOv2 outputs a tuple of 8 tensors, to which we assign names such as `FM_INTERMEDIATE_FEATURES_3`, `FM_INTERMEDIATE_CLS_TOKEN_3`, `FM_OUTPUT_FEATURES`, `FM_OUTPUT_CLS_TOKEN`. This way each model head can specify which tensors does it need. The names of all keys used for data transfered should be declared in `src/engine/naming_convention.py`.

## How to use
*In this section you will learn how to set up the engine and use it as a black box module.*

> [!NOTE]  
> Visual Perception Engine was developed and tested on Jetson Orin AGX with Jetpack 6.1. It should (not tested) work on other Jetson devices as long as jetpack version is 6.1 or higher.

First, you will have to clone the repository:
```
git clone https://github.com/nasa-jpl/visual-perception-engine

```

If you want to use ROS2 node be careful where you clone it. For more info see [the section on building ROS2 node](#ros2-node).

### Set up docker
For portability we used docker. Our environment is based on great work by [Dustin Franklin](https://github.com/dusty-nv). Firstly, export the following variables:
```bash
# copy your user into the docker
export _UID=$(id -u)
export _GID=$(id -g)

# path to workspace (relative to /home/${USER}) to be mounted inside docker
# if you are using only github repo
export WORKSPACE="path/to/cloned/repo"
# if you intend to use ros2 package
export WORKSPACE="path/to/ros2/workspace"
```
 Secondly, navigate to the `docker/` directory and then run these commands:
```bash
docker compose -f docker-compose.yml build # add `--no-cache` at the end to build from scratch
docker compose -f docker-compose.yml up -d
```
After these are completed run the following commands inside the container:
```bash
sudo git clone https://github.com/NVIDIA-AI-IOT/torch2trt /opt/torch2trt
cd /opt/torch2trt
sudo pip3 install .
cd - # to return to previous directory
```

### Install as package
Next, for simplicity of use install the package inside the container. To do so make sure that you are in the directory containing `pyproject.toml`. Then type:
```bash
python3 -m pip install .

# Or if you intend to modify the package internals use
python3 -m pip install -e .
```

Now you can verify that the package was successfully installed by running `pip show vp_engine`.

### Preparing model checkpoints
To run the default version of the engine first you will have to download all the necessary checkpoints from [here](https://drive.google.com/drive/folders/1SWMlEqOE_7EWPCkMloDTXG1_mZAmeW3-) and place them into `models/checkpoints/` folder. Once there run this command:
```bash
python3 -c "import vp_engine; vp_engine.export_default_models()"
```
This will export all the PyTorch models to TensorRT engines (stored in `models/engines` directory) and register all the models (i.e. add them to registry file `model_registry/registry.jsonl`) such that they can be easily loaded into the engine with desired parameters (e.g. precision)

> [!NOTE]  
> This step usually takes some time. You can expect up to 30 min of waiting.

### Set CUDA MPS
The engine uses multiple processes each using the same GPU. To make it possible you need to enable CUDA MPS. It is available for Tegra devices from CUDA 12.5. To enable it do:

```bash
#As $UID, run the commands

export CUDA_VISIBLE_DEVICES=0 # Select GPU 0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps/log

# make sure that the directories exist and have correct permissions
sudo mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
sudo chown $USER $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# Start the daemon
nvidia-cuda-mps-control -d

#This will start the MPS control daemon that will spawn a new MPS Server instance for that $UID starting an application and associate it with GPU visible to the control daemon.

# In case you want to disable CUDA MPS
echo quit | nvidia-cuda-mps-control
```


### Engine configuration
Engine was made to be easily configurable. You can write your own configuration files based on your needs and use it with the engine. The config should be in `json` format and follow the schema defined in `schemas/vp_engine_config.json`. In `configs/` there is already a `default.json` configuration file which specifies default configuration with 3 model heads.

In each configuration file one needs to specify the name of desired foundation models/model heads as specified in the model registry (cannonical name). Additionally, one can specify an alias that will be used across the engine instead of the lengthy cannonical name. For foundation model, you can specify a preprocessing function that you want to use, and for each model head you can specify a postprocessing function. Lastly, for each model you can specify rate, which is the upperbound on the inference frequency of each model (i.e. model can run slower in unexpected cases but it will not run faster than specified value).

Other parameters:
* `log_to_consol` specifies whether the log will be visible in the console
* `logging_level` specifies the level of information during logging
* `log_to_file` specifies the path to file where the log will be written, if not provided no log will be saved
* `canonical_image_shape_hwc` specifies the shape of images used as input to the engine
* `queue_sizes` specifies how many slots individual queues and buffers have
* `output_dir` is a optional parameter that is used with some internal testing functions

### Launching the engine
Once all of the above is completed you can use the engine. Below is an example code snippet:
```python
from vp_engine import Engine

engine = Engine()

# start all the processes and establishes interprocess communication
engine.build()

# send start signal all models are waiting for
engine.start_inference()

# send 3 dummy inputs to test and warmup the engine, HIGHLY RECOMMENDED
was_success: bool = engine.test() 

### engine is READY
```

> [!CAUTION]
> Once you do not need the engine anymore call `engine.stop()`. This will make sure to correctly close all the child processes. Without it there might be some processes hanging. Furthermore, if the engine crashed (Using `CTRL+C` is fine), there might be some processes hanging as well.

Now for basic functionality:

```python
# get an image in torch or numpy format
img = torch.zeroes(WIDTH, HEIGHT, N_CHANNELS).to(torch.uint8)

# to insert the engine for processing
was_success: bool = engine.input_image(img_torch) 

# get info about all the models
fm_params: dict = engine.get_foundation_model_params()
mh_params: list[dict] = engine.get_model_heads_params()
# In the parameters you can see what type of output to expect
# Currently it can be either an image or object detection result (labels+scores+normalized_boxes)

# check if the output of head number X is ready
X = 0 # as an example
output: None | np.ndarray | list[np.ndarray] = engine.get_head_output(X)

# to change the firing rate of any model at runtime:
new_rate = 10 #Hz
was_success: bool = change_model_rate("Model_name_to_target", new_rate)
```

> [!NOTE]  
> The engine class and its methods can be only called from within the process it was started in. Engine is not designed to be moved across processes.

### ROS2 node
If you want to use provided ROS2 (Humble) node, you will have to build it first. To do so you should have a ROS2 workspace directory set up (e.g. `~/ros2_ws`), in which you should have `src` directory containing the source code of all your packages. Ideally, this repository should be in that `src` directory (e.g. `~/ros2_ws/src/visual-perception-engine`). Then, navigate to ROS2 workspace directory and run the following commands:
```bash
source /opt/ros/humble/install/setup.bash # set up ROS2 underlay, i.e. be able to use ros2 from command line
colcon build --packages-select vp_engine
source install/setup.bash
```
> [!NOTE]  
> The core files for the node can be found in `ros_node/` or `include/vp_engine/` directories.

#### Usage
Once the package is built you can launch it using:
```bash
ros2 launch vp_engine engine_launch.xml
```
The launch file `launch/engine_launch.xml` contains several parameters that you can adjust as needed, for example the topic name from which the images should be taken.

The node will create a publisher for every model head listed in the configuration, using alias as a topic name (or `cannonical_name` if alias was not provided).

The node offers two services: `GetModelNames` - which returns list of the names of all models inside the engine, and `ChangeModelRate`- which allows the user to change firing rate of a particular model at runtime.

## For developers
*In this section you will learn how the engine works internally and how to extend it with your own models.*

If you would like to contribute check out [CONTRIBUTING.md](CONTRIBUTING.md)

### Adding new models
1. Add your implementation to `src/model_architectures`. Whether it is a backbone or a model head make sure to implement `ModelInterfaceBase`. Make sure that you implement `input_signature` and `output_signature` properties correctly. There are shapes checks throughout the engine so you need to know what kind of inputs and outputs you are expecting.
2. Add the class name to `__all__` in `src/model_architectures/__init__.py`. Make sure to correctly import the class.
3. Use `ModelExporter` to initialize the model. This way it will be converted to correct precision/framework and it will be registered in `ModelRegistry`, so that it will be easier to use later on.
4. Once your model is in the model registry you can simply add it to the configuration file and it's ready to use!

#### Adding new preprocessing/postprocessing functions
The process here is similar to the one for models. However, here you need to implement either `AbstractPreprocessing` or `AbstractPostprocessing`, and add the class name to `__all__` in `src/transforms/__init__.py`.

### How all of this works internally
All models within the engine (foundation_models and model_heads) implement `ModelInterfaceBase` to make the tensor exchange between individual models easier. For example, tensors are packed into a dict when moved between different models. This allows model A to output a set of different feature tensors such that model B can flawlessly take a subset of these features. A highlevel functionality `forward_annotated` is exposed, which takes dict as an input and outputs a dict, allows users to differctly stack models on top of eachother. 

However, each model is internally build around `forward` call which can take arguments (and output results) of any shape their creators' hearts desired. This combined with the fact that ONNX (our implementation first converts models to ONNX and only then to TensorRT) does not support dicts resulted in `forward_annotated` being a wrapper of `forward` instead of drop-in replacement as originally planned. Furhtermore, `ModelInterfaceBase` requires one to implement `deannotate_input` and `annotate_output` methods. First function converts the input dict to the format accepted by `forward`, and the latter converts the output of `forward` to a dict. 

**Important:** For proper functioning ensure that dict keys match across different models. E.g. make sure that when you implement a new model head its `input_signature` will have keys that are in `output_signature` of the foundation model.

## Authors
If you have any questions please reach out to any of the authors or open a github issue.
* Jakub Łucki
* Jonathan Becktor 
* Shehryar Khattak (skhattak@jpl.nasa.gov)
* Rob Royce (rob.royce@jpl.nasa.gov)
