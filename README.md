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
- [TPU compiler](https://coral.ai/docs/edgetpu/compiler/)

## 1) TensorFlow Lite
This framework is directly integrated in Tensorflow and allows to optimize TF and Keras models for fast inference on mobile, embedded, and IoT devices. It is composed on two main components:
- __TF Lite Converter__ to convert the models
- __TF Lite Interpreter__ to run inference with converted models

The conversion is done with the Python API. TensorFlow Lite currently supports a [limited subset of TensorFlow operations](https://www.tensorflow.org/lite/guide/ops_compatibility). If a model has unsupported operations, it is possible to force the converter to use the orignal TF operations, resulting in a less efficient but still successfull conversion. 
TF Lite allows for different types of conversion:
1. __No quantization__: the model is converted with some optimization operation (e.g. pruning of training-related nodes), weights and activations are stored as float32 numbers.
2. __Float16 quantization__: reduces model size by up to half (since all weights are now half the original size) with minimal loss in accuracy. Can speed up processing with GPUs.
3. __Weight quantization__: quantizes *only the weights* from floating point to 8-bits integers, reducing the model size up to 4x and speeding up inference. During inference some operations will be executed with integer kernel, others with float kernel (*hybrid operators*).
4. __Integer quantization__: all model values (weights and activations) are quantized to 8-bit integers. This results in a 4x reduction in model size and a 3 to 4x performance improvement on CPU performance. It needs a rapresentative part of the dataset to qunatize activations. If all the operations are supported it results in a __full integer quantization__, compatible with some hardware accelartors (e.g. Coral). Otherways the incompatible operations fall back in float32.

In options 2 to 4, __post-training quantization__ of weights and/or activations is performed. This allows to drastically reduce the model binary size and can speed up inference depending on the chosen inference hardware (CPU/GPU/TPU). Of course reducing the model size and increading the inference speed result in a loss of accuracy. A good conversion process looks for the best size/speed/accuracy tradeoff, depending on the specific application. A quantization-aware training is also possible to consider the quantization already during the training process, resulting in better accuracy (currently not supported by TF2).  
For more informations about TF Lite refer to the [official guide](https://www.tensorflow.org/lite/guide).
