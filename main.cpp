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

#include "utils.hpp"


int main(int, char **)
{
    cv::Mat vm = cv::imread("../4.jpg");

    std::unique_ptr<tflite::FlatBufferModel> model_uptr = tflite::FlatBufferModel::BuildFromFile("../best-fp16.tflite");
    assert(model_uptr != nullptr);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_uptr.get(), resolver);
    std::unique_ptr<tflite::Interpreter> interpreter_uptr;
    assert(kTfLiteOk == builder(&interpreter_uptr));

    interpreter_uptr->SetAllowFp16PrecisionForFp32(true);
    interpreter_uptr->AllocateTensors();
    interpreter_uptr->SetNumThreads(2);

    // tflite::PrintInterpreterState(interpreter_uptr.get());

    int In = interpreter_uptr->inputs()[0];
    int model_height = interpreter_uptr->tensor(In)->dims->data[1];
    int model_width = interpreter_uptr->tensor(In)->dims->data[2];
    int model_channels = interpreter_uptr->tensor(In)->dims->data[3];

    std::cout << "height   : " << model_height << std::endl;
    std::cout << "width    : " << model_width << std::endl;
    std::cout << "channels : " << model_channels << std::endl;

    std::cout << "tensors size : " << interpreter_uptr->tensors_size() << "\n";
    std::cout << "nodes size   : " << interpreter_uptr->nodes_size() << "\n";
    std::cout << "inputs       : " << interpreter_uptr->inputs().size() << " : " << interpreter_uptr->GetInputName(0) << "\n";
    std::cout << "outputs      : " << interpreter_uptr->outputs().size() << " : " << interpreter_uptr->GetOutputName(0) << "\n";

    TfLiteIntArray *output_dims = interpreter_uptr->tensor(interpreter_uptr->outputs()[0])->dims;
    auto output_size = output_dims->data[output_dims->size - 2];
    std::cout << "outputs size : "<< output_size << std::endl;

    float (*res_r)[6] = new float[output_size][6];

    std::vector<BboxWithScore> res_t;
    res_t.resize(output_size);
    while (true)
    {
        vm = cv::imread("../4.jpg");
        assert(!vm.empty());
        cv::resize(vm, vm, cv::Size(model_height, model_width));
        
        for (size_t h = 0; h < model_height; h++)
            for (size_t w = 0; w < model_width; w++)
            {
                interpreter_uptr->typed_input_tensor<float>(0)[h*model_width*model_channels + w*model_channels + 0] = vm.at<cv::Vec3b>(h,w)[0]/255.0;
                interpreter_uptr->typed_input_tensor<float>(0)[h*model_width*model_channels + w*model_channels + 1] = vm.at<cv::Vec3b>(h,w)[1]/255.0;
                interpreter_uptr->typed_input_tensor<float>(0)[h*model_width*model_channels + w*model_channels + 2] = vm.at<cv::Vec3b>(h,w)[2]/255.0;
            }
        
        auto t_start = std::chrono::high_resolution_clock::now();
        interpreter_uptr->Invoke();
        auto t_end = std::chrono::high_resolution_clock::now();

        memcpy((void*)res_r, interpreter_uptr->typed_output_tensor<float>(0), 6*output_size*sizeof(float));
        
        for (int ttt = 0; ttt < output_size; ttt++)
            if (res_r[ttt][4] > 0.3)
            {
                 // std::cout<<std::endl;
                res_t.push_back((BboxWithScore){
                        .tx = res_r[ttt][0]-res_r[ttt][2]/2, 
                        .ty = res_r[ttt][1]-res_r[ttt][3]/2, 
                        .bx = res_r[ttt][0]+res_r[ttt][2]/2, 
                        .by = res_r[ttt][1]+res_r[ttt][3]/2, 
                        .area = res_r[ttt][2]*res_r[ttt][3], 
                        .score = res_r[ttt][4]
                    });
            }

        softNms(res_t, 1, 0.5, 0.4, 0.001);

        for (auto &bx_s: res_t)
        {
            cv::rectangle(vm, cv::Point(int(bx_s.tx*640), int(bx_s.ty*640)), cv::Point(int(bx_s.bx*640), int(bx_s.by*640)), cv::Scalar(0,255,0));
        }
        cv::imshow("11", vm);
        cv::waitKey(500);


        auto t_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        std::cout << "infer used:" << t_dur << " ms" << std::endl;
    }
    delete 
}
