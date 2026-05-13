#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "HARmodel_wrapper.h"
#include <iomanip> 

double calcMean(const std::vector<double>& v)
{
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double calcStd(
    const std::vector<double>& v,
    double mean)
{
    double sum = 0.0;

    for (double x : v)
    {
        sum += (x - mean) * (x - mean);
    }

    return std::sqrt(sum / v.size());
}

double calcEnergy(const std::vector<double>& v)
{
    double e = 0.0;

    for (double x : v)
    {
        e += x * x;
    }

    return e / v.size();
}

std::vector<double> computeFFTFeatures(
    const std::vector<double>& signal)
{
    std::vector<double> features;

    int N = static_cast<int>(signal.size());

    // convert vector -> cv::Mat
    cv::Mat input(signal);

    input.convertTo(input, CV_64F);

    // FFT
    cv::Mat fft_result;

    cv::dft(
        input,
        fft_result,
        cv::DFT_COMPLEX_OUTPUT
    );

    // equivalent to rfft
    int fft_size = N / 2 + 1;

    std::vector<double> magnitudes;

    for (int i = 0; i < fft_size; i++)
    {
        double real =
            fft_result.at<cv::Vec2d>(i)[0];

        double imag =
            fft_result.at<cv::Vec2d>(i)[1];

        double mag =
            std::sqrt(real * real + imag * imag);

        magnitudes.push_back(mag);
    }

    // FFT mean
    double fft_mean = calcMean(magnitudes);

    // FFT std
    double fft_std =
        calcStd(magnitudes, fft_mean);

    // dominant frequency
    int dominant_freq =
        static_cast<int>(
            std::max_element(
                magnitudes.begin(),
                magnitudes.end()
            ) - magnitudes.begin()
        );

    // spectral energy
    double spectral_energy = 0.0;

    for (double x : magnitudes)
    {
        spectral_energy += x * x;
    }

    spectral_energy /= magnitudes.size();

    // spectral entropy
    double power_sum = 0.0;

    for (double x : magnitudes)
    {
        power_sum += x * x;
    }

    double entropy = 0.0;

    for (double x : magnitudes)
    {
        double psd =
            (x * x) / (power_sum + 1e-6);

        entropy -=
            psd * std::log(psd + 1e-6);
    }

    features.push_back(fft_mean);
    features.push_back(fft_std);
    features.push_back(dominant_freq);
    features.push_back(spectral_energy);
    features.push_back(entropy);

    return features;
}

std::vector<double> extract_features(
    const cv::Mat& sample)
{
    // sample:
    // rows = 128
    // cols = 9
    // type = CV_64F

    std::vector<double> all_features;

    int rows = sample.rows;
    int cols = sample.cols;

    for (int j = 0; j < cols; j++)
    {
        std::vector<double> channel(rows);

        for (int i = 0; i < rows; i++)
        {
            channel[i] =
                sample.at<double>(i, j);
        }

        // ===== Time-domain =====

        double mean_val =
            calcMean(channel);

        double std_val =
            calcStd(channel, mean_val);

        double max_val =
            *std::max_element(
                channel.begin(),
                channel.end()
            );

        double min_val =
            *std::min_element(
                channel.begin(),
                channel.end()
            );

        double energy =
            calcEnergy(channel);

        all_features.push_back(mean_val);
        all_features.push_back(std_val);
        all_features.push_back(max_val);
        all_features.push_back(min_val);
        all_features.push_back(energy);

        // ===== Frequency-domain =====

        std::vector<double> fft_features =
            computeFFTFeatures(channel);

        all_features.insert(
            all_features.end(),
            fft_features.begin(),
            fft_features.end()
        );
    }

    return all_features;
}

cv::Mat ReadData(std::string FileName)
{
    cv::Mat inputMat(128,9,CV_64F);
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
        if (lineStr.empty()) continue; // 跳過空行
        if (index >= 128 * 9) break;   // 防止超過預設大小
        int row = index / 9;
        int col = index % 9;
        //std::cout<< lineStr <<std::endl;
        inputMat.at<float>(row,col)= std::stod(lineStr);
        index+=1;

    }
    return inputMat;
  
}


int TreeModelPrediction(std::vector<double> feature)
{
    std::vector<double> predictions(6, 0.0);
    classify_action(feature.data(), predictions.data());
    int best_class = 0;
    double max_prob = -1.0;

    for (int i = 0; i < predictions.size(); ++i) {
        if (predictions[i] > max_prob) {
            max_prob = predictions[i];
            best_class = i;
        }
    }

    std::cout << "預測的動作類別 ID: " << best_class << " (概率: " << max_prob << ")" << std::endl;
    return best_class;
}

int main()
{

    std::string FileName = "/home/zonekey/project/action_onnx/cpp_test_samples/sample_1.txt";
    cv::Mat inputMat = ReadData(FileName);
    std::vector<double>inputFeatures = extract_features(inputMat);
    int predclass = TreeModelPrediction( inputFeatures);

    return 0;
}