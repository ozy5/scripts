#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;

// here use this bro 
// g++ filter_experiment.cpp -o filter_experiment `pkg-config --cflags --libs opencv4`

void process_image(const String& img_path, const String& output_path, const vector<uchar>& LUT_blue, const vector<uchar>& LUT_red, Ptr<CLAHE> clahe)
{
    Mat img = imread(img_path, IMREAD_COLOR);

    // print the shape of image
    cout << "Image shape: " << img.size() << endl;

    vector<Mat> channels;
    split(img, channels);
    auto start = chrono::high_resolution_clock::now();

    // apply CLAHE to first channel
    clahe->apply(channels[0], channels[1]);

    // apply LUTs to other channels
    LUT(channels[1], LUT_blue, channels[0]);
    LUT(channels[1], LUT_red, channels[2]);

    merge(channels, img);

    auto end = chrono::high_resolution_clock::now();
    double taken_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Time taken: " << taken_time << " ms" << endl;

    // save image with filtered name with original filename
    imwrite(output_path + "filtered" + img_path.substr(img_path.find_last_of("/")), img);

    // print progress and filename
    cout << "Processed: " << img_path << endl;
}

int main(int argc, char **argv)
{

    if (argc != 5)
    {
        cout << "Usage: " << argv[0] << " blue_start red_start image_path output_path" << endl;
        return -1;
    }

    // Parse command line arguments
    int blue_mapping_start_point = stoi(argv[1]);
    int red_mapping_start_point = stoi(argv[2]);
    String path = argv[3];
    String output_path = argv[4];

    // Define LUTs
    vector<uchar> LUT_blue(256);
    vector<uchar> LUT_red(256);
    // int blue_mapping_start_point = 180;
    int blue_mapping_slope = 255 / (255 - blue_mapping_start_point);
    // int red_mapping_start_point = 200;
    int red_mapping_slope = 255 / (255 - red_mapping_start_point);
    for (int i = 0; i < 256; i++)
    {
        LUT_blue[i] = (uchar)((i - blue_mapping_start_point) * blue_mapping_slope * (i > blue_mapping_start_point));
        LUT_red[i] = (uchar)((i - red_mapping_start_point) * red_mapping_slope * (i > red_mapping_start_point));
    }

    cv::Mat LUT_red_mat(256, 1, CV_8UC1, LUT_red.data());
    cv::Mat LUT_blue_mat(256, 1, CV_8UC1, LUT_blue.data());

    cout << LUT_red_mat.total() << endl;
    // Find all images in the folder
    vector<String> filenames;
    // String path = "/home/zado/datasets/HIT_UAV_and_NII_CU_dataset/train/images/";
    glob(path, filenames);

    // For each image
    vector<double> taken_times;
    Ptr<CLAHE> clahe = createCLAHE(2, Size(4, 4));
    // img will be a 640x640x3 matrix Cv 8UC3
    // create img variable with zeros

    int num_threads = thread::hardware_concurrency();
    cout << "Number of threads: " << num_threads << endl;

    auto start_time = chrono::high_resolution_clock::now();

    parallel_for_(Range(0, filenames.size()), [&](const Range& range) {
        for (int i = range.start; i < range.end; i++)
        {
            process_image(filenames[i], output_path, LUT_blue, LUT_red, clahe);
        }
    });

    auto end_time = chrono::high_resolution_clock::now();
    double total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Total time taken: " << total_time << " ms" << endl;
    // average time taken
    cout << "Average time taken: " << total_time / filenames.size() << " ms" << endl;
    return 0;
}