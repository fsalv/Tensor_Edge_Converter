# Tensor_Edge_Converter
This repository provide simple and ready-to-use Jupyter notebooks for converting Keras (TensorFlow 2) models for Edge Computing applications.

The optimization frameworks considered are:
- [TensorFlow Lite](https://www.tensorflow.org/lite) (Google): CPU/GPU inference
- [EdgeTPU](https://coral.ai/) (Google Coral): TPU inference
- [TensorRT](https://developer.nvidia.com/tensorrt) (NVIDIA): CPU/GPU inference

## 0) Prerequisites and installations
First clone this repository:

   ```bash
   git clone https://github.com/fsalv/Tensor_Edge_Converter.git
   ```
Jupyter Notebook and Python3 are required:
  
  ```bash
  sudo apt-get update
  sudo apt-get install pip3 -y
  pip3 install jupyter
   ```
The following libraries are needed for the conversions:
- [numpy](https://pypi.org/project/numpy/)
- [Tensorflow 2](https://www.tensorflow.org/install)
- [TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)
- [EdgeTPU compiler](https://coral.ai/docs/edgetpu/compiler/)

## 1) Repository usage
The different Jupyter notebooks allow to convert keras models according with the different frameworks presented.
Put the keras model to be converted inside __./model__, saved as _model_name.h5_. You will find the results of the conversion inside the same folder.

Some methods need a _representative_dataset_ to be used during conversion, in order to perform activations quantization correctly. This representative dataset is usually part of the training dataset and should theorethically cover all possible conditions in which the network could operate (e.g all the classes in a multi-class classification network). In this case a _computer vision_ application is taken in consideration, and a set of images are supposed to be put inside __./dataset__ folder. Feel free to change the code in order to manage different kinds of data.

A trivial non-trained model (Sequential of Conv2D and FC) is provided to verify the conversion process, as well as some images from [Kaggle's flowers classification dataset](https://www.kaggle.com/alxmamaev/flowers-recognition) (in particular, the 734 images of the sunflower class).



## 2) TensorFlow Lite
This framework is directly integrated in Tensorflow and allows to optimize TF and Keras models for fast inference on mobile, embedded, and IoT devices. It is composed on two main components:
- __TF Lite Converter__ to convert the models
- __TF Lite Interpreter__ to run inference with converted models

The conversion is done with the Python API. TensorFlow Lite currently supports a [limited subset of TensorFlow operations](https://www.tensorflow.org/lite/guide/ops_compatibility). If a model has unsupported operations, it is possible to force the converter to use the orignal TF operations, resulting in a less efficient but still successfull conversion. 
TF Lite has four different types of conversion:
1. __No quantization__: the model is converted with some optimization operation (e.g. pruning of training-related nodes), weights and activations are stored as float32 numbers.
2. __Float16 quantization__: reduces model size by up to half (since all weights are now half the original size) with minimal loss in accuracy. Model still executes as float32 operations. Can speed up processing with GPUs.
3. __Weight quantization__: quantizes *only the weights* from floating point to 8-bits integers, reducing the model size up to 4x and speeding up inference. During inference some operations will be executed with integer kernel, others with float kernel (*hybrid operators*).
4. __Integer quantization__: all model values (weights and activations) are quantized to 8-bit integers. This results in a 4x reduction in model size and a 3 to 4x performance improvement on CPU performance. It needs a representative part of the dataset to quantize activations. If all the operations are supported it results in a __full integer quantization__, compatible with some hardware accelartors (e.g. Coral). Otherways the incompatible operations fall back in float32.

In options 2 to 4, __post-training quantization__ of weights and/or activations is performed. This allows to drastically reduce the model binary size and can speed up inference depending on the chosen inference hardware (CPU/GPU/TPU). Of course, reducing the model size and increading the inference speed result in a loss of accuracy. A good conversion process looks for the best size-speed-accuracy tradeoff, depending on the specific application. A quantization-aware training is also possible to already consider the quantization during the training process, resulting in better accuracy (currently not supported by TF2).  

For more informations about TF Lite refer to the [official guide](https://www.tensorflow.org/lite/guide).
For inference with tflite models, [TFLite Interpreter](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python) should be used. In the future, I plan to provide minimal code for inference, as well.

## 3) EdgeTPU
To make a TF model compatible with Coral Edge TPU, a TFLite __full integer quantization__ is needed. Read point _4_ of the previous section for how it works.

After the TF Lite conversion, the __Edge TPU Compiler__ has to be used to get the final model. The output of this final conversion is still a tflite model, but can be deployed on the Coral Dev Board or the Coral USB Accelerator. Only a subset of TFLite-supported operations can be then succesfull executed on EdgeTPU hardware. Unsupported operations can be still mapped to CPU, with significant loss of performance. See [here](https://coral.ai/docs/edgetpu/models-intro/#supported-operations), for currently supported ops. Note that different versions of compiler and runtime version are available, with different compatibility with TF operations.

The inference process is pretty much the same as for normal tflite models, but the Edge TPU runtime library should be passed as delegate when the Interpreter object is instantiated. More informations on inference can be found on [Coral website](https://coral.ai/docs/edgetpu/tflite-python/). <br>
Note that the runtime library should be installed on the system as follows:
- __USB Accelerator__: follow instructions [here](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime) to install the binary on the host system. Note that you can choose between the standard runtime, and the one that operates at maximum operating frequency. The latter increases inference speed, but with increased power consumption and a significant raise of temperature of the USB Accelerator.
- __Dev Board__: follow the [Get Started guide](https://coral.ai/docs/dev-board/get-started/) to flash the board and you will end up with a system with all the necessary libraries installed.

## 4) TensorRT
[Coming soon]