#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "action_model_wrapper.h"
#include <iomanip> // 用於格式化輸出

void PrintFullReport(const std::vector<int>& labels, const std::vector<int>& predictions, int num_classes) {
    // 1. 初始化混淆矩陣 (N x N)
    std::vector<std::vector<int>> matrix(num_classes, std::vector<int>(num_classes, 0));

    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] >= 0 && labels[i] < num_classes && predictions[i] >= 0 && predictions[i] < num_classes) {
            matrix[labels[i]][predictions[i]]++;
        }
    }

    std::vector<std::string> class_names = {"Walk", "StandUp", "SitDown", "StandStill", "SitStill"};

    // 2. 打印混淆矩陣
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "CONFUSION MATRIX\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::setw(15) << "Actual \\ Pred";
    for (const auto& name : class_names) std::cout << std::setw(12) << name;
    std::cout << "\n";

    for (int i = 0; i < num_classes; ++i) {
        std::cout << std::setw(15) << class_names[i];
        for (int j = 0; j < num_classes; ++j) {
            std::cout << std::setw(12) << matrix[i][j];
        }
        std::cout << "\n";
    }

    // 3. 計算 Precision 與 Recall
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << std::setw(15) << "Class" << std::setw(15) << "Precision" << std::setw(15) << "Recall" << "\n";
    std::cout << std::string(50, '-') << "\n";

    double total_accuracy = 0;
    int total_samples = labels.size();

    for (int i = 0; i < num_classes; ++i) {
        int tp = matrix[i][i]; // True Positive
        
        // Recall: 當前行的總和 (所有真正的該類別)
        int row_sum = 0;
        for (int j = 0; j < num_classes; ++j) row_sum += matrix[i][j];
        
        // Precision: 當前列的總和 (所有模型預測為該類別的)
        int col_sum = 0;
        for (int j = 0; j < num_classes; ++j) col_sum += matrix[j][i];

        double precision = (col_sum > 0) ? (double)tp / col_sum : 0.0;
        double recall = (row_sum > 0) ? (double)tp / row_sum : 0.0;

        std::cout << std::setw(15) << class_names[i] 
                  << std::setw(14) << std::fixed << std::setprecision(2) << precision * 100 << "%"
                  << std::setw(14) << recall * 100 << "%\n";
        
        total_accuracy += tp;
    }

    std::cout << std::string(50, '-') << "\n";
    std::cout << "Overall Accuracy: " << (total_accuracy / total_samples) * 100 << "%\n";
    std::cout << std::string(50, '=') << "\n";
}

double calculate_std(const std::vector<double>& v, double mean) {
    double sum = 0;
    for (double x : v) sum += (x - mean) * (x - mean);
    return std::sqrt(sum / v.size());
}

double calculate_slope(const std::vector<double>& y) {
    int n = y.size();
    if (n < 2) return 0;
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < n; i++) {
        sum_x += i;
        sum_y += y[i];
        sum_xy += i * y[i];
        sum_xx += i * i;
    }
    double denominator = (n * sum_xx - sum_x * sum_x);
    if (std::abs(denominator) < 1e-9) return 0;
    return (n * sum_xy - sum_x * sum_y) / denominator;
}


std::vector<double> ExtractFeatures(cv::Mat sampleArray) {
    int n = sampleArray.rows; // 18
    std::vector<double> body_w(n), body_h(n), aspect_ratio(n);
    std::vector<double> head_center_y(n), face_relative_pos(n);
    std::vector<double> center_x(n), center_y(n), scores(n);
    std::vector<int> actIDs(n);

    // 1. 基礎數據提取與初步計算
    for (int i = 0; i < n; i++) {
        double ax1 = sampleArray.at<double>(i, 0), ay1 = sampleArray.at<double>(i, 1);
        double ax2 = sampleArray.at<double>(i, 2), ay2 = sampleArray.at<double>(i, 3);
        double hx1 = sampleArray.at<double>(i, 4), hy1 = sampleArray.at<double>(i, 5);
        double hx2 = sampleArray.at<double>(i, 6), hy2 = sampleArray.at<double>(i, 7);
        
        body_w[i] = ax2 - ax1;
        body_h[i] = ay2 - ay1;
        aspect_ratio[i] = body_h[i] / (body_w[i] + 1e-6);
        
        center_x[i] = hx2 - hx1; 
        center_y[i] = hy2 - hy1;
        
        double h_center_y = (hy1 + hy2) / 2.0;
        head_center_y[i] = h_center_y;
        face_relative_pos[i] = (h_center_y - ay1) / (body_h[i] + 1e-6);
        
        actIDs[i] = (int)sampleArray.at<double>(i, 8);
        scores[i] = sampleArray.at<double>(i, 9);
    }

    // --- 開始構建特徵向量 ---
    std::vector<double> f;

    // 1. 身體比例統計
    double ar_mean = std::accumulate(aspect_ratio.begin(), aspect_ratio.end(), 0.0) / n;
    f.push_back(ar_mean); // ratio_mean
    f.push_back(calculate_std(aspect_ratio, ar_mean)); // ratio_std
    f.push_back(*std::max_element(aspect_ratio.begin(), aspect_ratio.end())); // ratio_max

    // 2. 空間穩定性
    double frp_mean = std::accumulate(face_relative_pos.begin(), face_relative_pos.end(), 0.0) / n;
    f.push_back(frp_mean); // face_pos_mean
    f.push_back(calculate_std(face_relative_pos, frp_mean)); // face_pos_std

    // 總移動距離
    double total_dist_x = 0, total_dist_y = 0, max_step_dist = 0;
    for(int i=0; i<n-1; i++) {
        double dx = center_x[i+1] - center_x[i];
        double dy = center_y[i+1] - center_y[i];
        total_dist_x += std::abs(dx);
        total_dist_y += std::abs(dy);
        max_step_dist = std::max(max_step_dist, dx*dx + dy*dy);
    }
    f.push_back(std::sqrt(total_dist_x * total_dist_x + total_dist_y * total_dist_y)); // total_movement
    f.push_back(max_step_dist); // max_step_dist

    // 3. 運動趨勢
    double h_slope = calculate_slope(body_h);
    double h_mean = std::accumulate(body_h.begin(), body_h.end(), 0.0) / n;
    f.push_back(h_slope); // height_slope
    f.push_back(calculate_std(body_h, h_mean)); // height_std

    // 4. 上游信息
    int standing_count = 0;
    for(int id : actIDs) if(id == 1) standing_count++;
    f.push_back((double)standing_count); // vote_count
    f.push_back(std::accumulate(scores.begin(), scores.end(), 0.0) / n); // avg_score
    f.push_back(*std::min_element(scores.begin(), scores.end())); // min_score

    // 5. 狀態切換
    int flips = 0;
    for(int i=0; i<n-1; i++) {
        if((actIDs[i] == 1) != (actIDs[i+1] == 1)) flips++;
    }
    f.push_back((double)flips); // state_flips

    // A. 分段統計
    double first_third_mean = std::accumulate(aspect_ratio.begin(), aspect_ratio.begin()+6, 0.0) / 6.0;
    double last_third_mean = std::accumulate(aspect_ratio.begin()+12, aspect_ratio.end(), 0.0) / 6.0;
    f.push_back(first_third_mean); // ratio_first_third
    f.push_back(last_third_mean);  // ratio_last_third
    f.push_back(last_third_mean - first_third_mean); // ratio_trend_diff

    // B. 斜率擴展
    f.push_back(calculate_slope(aspect_ratio)); // ratio_slope
    f.push_back(h_slope); // height_slope (重復了，但需對齊Python)
    f.push_back(std::abs(h_slope)); // slope_abs

    // C. 速度與抖動
    std::vector<double> velocity;
    for(int i=0; i<n-1; i++) velocity.push_back(aspect_ratio[i+1] - aspect_ratio[i]);
    double v_mean = std::accumulate(velocity.begin(), velocity.end(), 0.0) / velocity.size();
    f.push_back(v_mean); // velocity_mean
    f.push_back(calculate_std(velocity, v_mean)); // velocity_std

    // D. 峰值時間
    f.push_back((double)std::distance(aspect_ratio.begin(), std::max_element(aspect_ratio.begin(), aspect_ratio.end()))); // peak_timing

    // E. 概率演變
    f.push_back(calculate_slope(scores)); // confidence_trend

    return f;
}

int TreeModelPrediction(std::vector<double> feature)
{
    std::vector<double> predictions(5, 0.0);
    classify_action(feature.data(), predictions.data());
    int best_class = 0;
    double max_prob = -1.0;

    for (int i = 0; i < predictions.size(); ++i) {
        if (predictions[i] > max_prob) {
            max_prob = predictions[i];
            best_class = i;
        }
    }

    //std::cout << "預測的動作類別 ID: " << best_class << " (概率: " << max_prob << ")" << std::endl;
    return best_class;
}
void printMatFull(const cv::Mat& m, const std::string& name)
{
    std::cout << "===== Matrix: " << name << " =====" << std::endl;
    std::cout << "Size: " << m.rows << " x " << m.cols << std::endl;

    for (int i = 0; i < m.rows; i++) {
        std::cout << "[" << i << "]: ";
        for (int j = 0; j < m.cols; j++) {
            std::cout << m.at<float>(i, j);
            if (j + 1 < m.cols) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "====================================" << std::endl;
}


int main()
{

    std::ifstream inFile("/home/zonekey/video_analyse/libtea/build_pc/test.csv",std::ios::in);
    if (!inFile.is_open()) return -1;
    std::string lineStr, header;
    // skip the headline
    std::getline(inFile, header);

    cv::Mat sampleArray(18,10,CV_64F);
    std::vector<int>labels;
    std::vector<int>predictions;
 


    int line_count=0;
    while (std::getline(inFile,lineStr))
    {
        std::stringstream ss(lineStr);
        std::string cell;
       

        if (line_count < 18)
        {
            //skip the time stamp firstly
            std::getline(ss, cell, ',');
            // skip ID secondly
            std::getline(ss, cell, ',');
            int col_num = 0;
            while (std::getline(ss, cell, ',') && col_num < 10) {
                sampleArray.at<double>(line_count, col_num) = std::stod(cell);
                col_num++;
            }
            line_count+=1;

        }
        else
        {
            // get the true label
            std::getline(ss, cell, ','); 
            labels.push_back(std::stoi(cell));

            // model prediction
            //printMatFull(sampleArray,"sample");
            std::vector<double> feature = ExtractFeatures(sampleArray);
            int pred = TreeModelPrediction(feature);
            predictions.push_back(pred);
            
            // reset the line count to prepare next sample
            line_count =0;

        }

       
       
    }
    // calculate the accuracy
    int correct = 0;
    for(size_t i=0; i<predictions.size(); ++i) {
        if(predictions[i] == labels[i]) correct++;
    }
    std::cout << "Accuracy: " << (float)correct/predictions.size() << std::endl;
    if (!labels.empty()) {
        PrintFullReport(labels, predictions, 5); // 假設有 5 個類別
    }

    

    





    return 0;
}











