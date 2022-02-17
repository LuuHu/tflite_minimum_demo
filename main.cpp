#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/opencv.hpp>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <memory>
#include <chrono>

int main(int, char**) {

    cv::VideoCapture cap("../../person.mp4");


    std::unique_ptr<tflite::FlatBufferModel> model_uptr =  tflite::FlatBufferModel::BuildFromFile("../yolov5s-fp16.tflite");
    assert(model_uptr!=nullptr);

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder builder(*model_uptr.get(), resolver);

    std::unique_ptr<tflite::Interpreter> interpreter_uptr;
    assert(kTfLiteOk == builder(&interpreter_uptr));
    
    interpreter_uptr->SetAllowFp16PrecisionForFp32(true);
    interpreter_uptr->AllocateTensors();
    interpreter_uptr->SetNumThreads(2);

    // tflite::PrintInterpreterState(interpreter_uptr.get());


    int In = interpreter_uptr->inputs()[0];
    int model_height   = interpreter_uptr->tensor(In)->dims->data[1];
    int model_width    = interpreter_uptr->tensor(In)->dims->data[2];
    int model_channels = interpreter_uptr->tensor(In)->dims->data[3];

    std::cout << "height   : "<< model_height << std::endl;
    std::cout << "width    : "<< model_width << std::endl;
    std::cout << "channels : "<< model_channels << std::endl;


    std::cout << "tensors size: " << interpreter_uptr->tensors_size() << "\n";
    std::cout << "nodes size: " << interpreter_uptr->nodes_size() << "\n";
    std::cout << "inputs: " << interpreter_uptr->inputs().size() << "\n";
    std::cout << "outputs: " << interpreter_uptr->outputs().size() << "\n";


    cv::Mat vm;

    while (true)
    {
        cap >> vm;
        assert(!vm.empty());
        cv::resize(vm, vm, cv::Size(model_height,model_width));

        std::cout<< int(*(vm.data+10))<<std::endl;

        // std::cout<<interpreter_uptr->inputs().size()<<std::endl;
        // cv::imshow("12", vm);
        // cv::waitKey(100);
        // // std::cout<<vm.total() * vm.elemSize()<<std::endl;
        // for (int i = 0; i < vm.total(); ++i) {
        //     const auto& rgb = vm[i];
        //     interpreter_uptr->typed_input_tensor<uchar>(0)[3*i + 0] = rgb[0];
        //     interpreter_uptr->typed_input_tensor<uchar>(0)[3*i + 1] = rgb[1];
        //     interpreter_uptr->typed_input_tensor<uchar>(0)[3*i + 2] = rgb[2];
        // }
        memcpy(interpreter_uptr->typed_input_tensor<float>(0), vm.data, 10);
        std::cout<<"cpy done\n";
        auto t_start = std::chrono::high_resolution_clock::now();
        interpreter_uptr->Invoke();
        auto t_end = std::chrono::high_resolution_clock::now();

        auto t_dur = std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start).count();
        std::cout<<"infer used:"<< t_dur<< " ms" << std::endl;

    
    }
    

}
