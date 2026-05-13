#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include<onnxruntime_cxx_api.h>

cv::Mat ReadData(std::string FileName)
{
    cv::Mat inputMat(128,9,CV_32F);
    std::ifstream inFile(FileName,std::ios::in);
    if (!inFile.is_open())
    {
        std::cerr << "File open failed: " << FileName << std::endl;
        return cv::Mat();
    }
    std::string lineStr;
    int index =0;
    while (std::getline(inFile,lineStr))
    {
        int row = index / 9;
        int col = index % 9;
        //std::cout<< lineStr <<std::endl;
        inputMat.at<float>(row,col)= std::stof(lineStr);
        index+=1;

    }
    return inputMat;
  
}

std::vector<float> Softmax(const std::vector<float>& logits)
{
    std::vector<float> probs(logits.size());

    // 1️⃣ 找最大值（數值穩定）
    float max_val = logits[0];
    for (size_t i = 1; i < logits.size(); i++)
    {
        if (logits[i] > max_val)
            max_val = logits[i];
    }

    // 2️⃣ 計算 exp(x - max)
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++)
    {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }

    // 3️⃣ normalize
    for (size_t i = 0; i < probs.size(); i++)
    {
        probs[i] /= sum;
    }

    return probs;
}


int InferenceProcess(cv::Mat inputMat,std::string onnx_path_name)
{
    // read model and get basic info
    std::ifstream infile(onnx_path_name);
    if (!infile.good()) {
        throw std::runtime_error("ONNX model file not found: " + onnx_path_name);
    }
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session = Ort::Session(env, onnx_path_name.c_str(), session_options);
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input information
    int time_step,feature_dim;
    for (size_t i = 0; i < numInputNodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();

        time_step = input_dims[1];
        feature_dim = input_dims[2];

        std::cout << "Input format = " << input_dims[0] << "x" << input_dims[1] << "x"
                  << input_dims[2] << std::endl;
    }

    // Get output information
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    std::cout << "Output format :  = " << output_dims[0] << "x" << output_dims[1] << std::endl;

    for (size_t i = 0; i < numOutputNodes; i++) {
        auto out_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }

    size_t tpixels = time_step*feature_dim;
    std::array<int64_t, 3> input_shape_info{1,time_step,feature_dim };
    // Create input tensor
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, inputMat.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

    const std::array<const char*, 1> inputNames = {input_node_names[0].c_str()};
    const std::array<const char*, 1> outNames = {output_node_names[0].c_str()};

    // Perform inference
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outNames.data(), outNames.size());
    } catch (const std::exception& e) {
        std::cerr << "Error during ONNX inference: " << e.what() << std::endl;
        return -1;
    }

    // Get model output
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<float> logits(pdata, pdata + output_dims[1]);
    std::vector<float>  probs = Softmax(logits);
    float max_prob=-100;
    int pred;
    for (int i=0;i< probs.size();i++)
    {
        float prob = probs[i];
        if (prob>max_prob)
        {
            max_prob= prob;
            pred =i;
        }
    }

    std::cout<<"prediction class is:" << pred << std::endl;
    std::cout<<"prediction score is:" << max_prob << std::endl;

    return pred;


}


int main()
{
    std::string FileName = "/home/zonekey/project/action_onnx/cpp_test_samples/sample_1.txt";
    cv::Mat inputMat = ReadData(FileName);
    std::string onnx_path_name ="/home/zonekey/project/action_onnx/lstm_classifier.onnx";
    InferenceProcess(inputMat,onnx_path_name);
    return 1;
}



