#include <cstdint>
#ifndef __ARM_NEON
// Use standard types for other platforms.
typedef float float16_t;
typedef double float64_t;
#endif
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <filesystem>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/timestamp.h>
}

const char* COUNTRY_CODES =
"AFG\nALB\nALG\nAND\nANG\nANT\nARG\nARM\nARU\nASA\nAUS\nAUT\nAZE\n"
"BAH\nBAN\nBAR\nBDI\nBEL\nBEN\nBER\nBHU\nBIH\nBIZ\nBLR\nBOL\nBOT\n"
"BRN\nBRU\nBUL\nBUR\nCAF\nCAM\nCAN\nCAY\nCGO\nCHA\nCHI\nCHN\nCIV\n"
"CMR\nCOD\nCOK\nCOL\nCOM\nCPV\nCRC\nCRO\nCUB\nCYP\nCZE\nDEN\nDJI\n"
"DMA\nDOM\nECU\nEGY\nERI\nESA\nESP\nEST\nETH\nFIJ\nFIN\nFRA\nFSM\n"
"GAB\nGAM\nGBR\nGBS\nGEO\nGEQ\nGER\nGHA\nGRE\nGRN\nGUA\nGUI\nGUM\n"
"GUY\nHAI\nHKG\nHON\nHUN\nINA\nIND\nIRI\nIRL\nIRQ\nISL\nISR\nISV\n"
"ITA\nIVB\nJAM\nJOR\nJPN\nKAZ\nKEN\nKGZ\nKIR\nKOR\nKSA\nKUW\nLAO\n"
"LAT\nLBA\nLBR\nLCA\nLES\nLIB\nLIE\nLTU\nLUX\nMAD\nMAR\nMAS\nMAW\n"
"MDA\nMDV\nMEX\nMGL\nMHL\nMKD\nMLI\nMLT\nMNE\nMON\nMOZ\nMRI\nMTN\n"
"MYA\nNAM\nNCA\nNED\nNEP\nNGR\nNIG\nNOR\nNRU\nNZL\nOMA\nPAK\nPAN\n"
"PAR\nPER\nPHI\nPLE\nPLW\nPNG\nPOL\nPOR\nPRK\nPUR\nQAT\nROU\nRSA\n"
"RUS\nRWA\nSAM\nSEN\nSEY\nSIN\nSKN\nSLE\nSLO\nSMR\nSOL\nSOM\nSRB\n"
"SRI\nSTP\nSUD\nSUI\nSUR\nSVK\nSWE\nSWZ\nSYR\nTAN\nTGA\nTHA\nTJK\n"
"TKM\nTLS\nTOG\nTPE\nTRI\nTUN\nTUR\nTUV\nUAE\nUGA\nUKR\nURU\nUSA\n"
"UZB\nVAN\nVEN\nVIE\nVIN\nYEM\nZAM\nZIM\n";

struct VideoAnalysisTouch
{
    uint32_t frame;
    uint8_t score1;
    uint8_t score2;
};

struct VideoAnalysis
{
    uint8_t touch_count;
    VideoAnalysisTouch touches[64];
};

struct StreamBoutSegment
{
    int32_t start_frame;
    int32_t end_frame;
    const char* name_left;
    const char* name_right;
    const char* country_left;
    const char* country_right;
    const char* tableau;
};

struct StreamAnalysis
{
    float64_t fps;
    int32_t frame_count;
    uint8_t bout_count;
    StreamBoutSegment bouts[64];
};

extern "C" {
    EXPORT int32_t add(int32_t a, int32_t b);
    VideoAnalysis* find_video_touches(const char* video_path, uint8_t overlay_id);
    EXPORT void js_memcpy(void* dest, void* source, size_t size);
    EXPORT void train_nn();
    // EXPORT StreamAnalysis* cut_stream(const char* tesseract_path, const char* svm_path, const char* video_path, uint8_t overlay_id, const char* output_folder, void (*callback)(int));
    EXPORT void cut_stream_async(const char* tesseract_path, const char* svm_path, const char* video_path, uint8_t overlay_id, const char* output_folder, const char* event_name, void (*callback)(int));
}

int32_t add(int32_t a, int32_t b) {
    return a + b;
}

struct VideoROI {
    float x;
    float y;
    float width;
    float height;

    void fromRect(cv::Rect rect, float scaleX, float scaleY)
    {
        x = rect.x / scaleX;
        y = rect.y / scaleY;
        width = rect.width / scaleX;
        height = rect.height / scaleY;
    }
};

std::ostream& operator<<(std::ostream& os, const VideoROI& obj) {
    os << "Video ROI: [" << obj.width << " x " << obj.height << " from (" << obj.width << ", " << obj.height << ")]";
    return os;
}

struct OverlayConfig {
    uint8_t id;
    float16_t threshold;
    bool symmetric_threshold;
    VideoROI red;
    VideoROI green;
    VideoROI red_score;
    VideoROI green_score;
    VideoROI red_name;
    VideoROI green_name;
    VideoROI red_country;
    VideoROI green_country;
    VideoROI time;
    VideoROI tableau;
};

std::ostream& operator<<(std::ostream& os, const OverlayConfig& obj) {
    os << "Overlay " << static_cast<int>(obj.id) << ": " << obj.red << ", " << obj.green << ", " << obj.red_score << ", " << obj.green_score;
    return os;
}

const OverlayConfig OVERLAY_STANDARD_1 = {
    .id = 0,
    .threshold = 0.171,
    .symmetric_threshold = false,
    .red = {.x = (float)228 / 1920, .y = (float)980 / 1080, .width = (float)608 / 1920, .height = (float)32 / 1080 },
    .green = {.x = (float)1063 / 1920, .y = (float)980 / 1080, .width = (float)608 / 1920, .height = (float)32 / 1080 },
    .red_score = {.x = (float)792 / 1920, .y = (float)927 / 1080, .width = (float)64 / 1920, .height = (float)48 / 1080 },
    .green_score = {.x = (float)1060 / 1920, .y = (float)927 / 1080, .width = (float)64 / 1920, .height = (float)48 / 1080 },
    .red_name = {.x = (float)338 / 1829, .y = (float)882 / 1031, .width = (float)410 / 1829, .height = (float)56 / 1031 },
    .green_name = {.x = (float)1100 / 1829, .y = (float)882 / 1031, .width = (float)410 / 1829, .height = (float)56 / 1031 },
    .red_country = {.x = (float)250 / 1829, .y = (float)891 / 1031, .width = (float)96 / 1829, .height = (float)36 / 1031 },
    .green_country = {.x = (float)1492 / 1829, .y = (float)891 / 1031, .width = (float)96 / 1829, .height = (float)36 / 1031 },
    .time = {.x = (float)869 / 1829, .y = (float)888 / 1031, .width = (float)108 / 1829, .height = (float)44 / 1031 }
};

const OverlayConfig OVERLAY_STANDARD_2 = {
    .id = 1,
    .threshold = 0.19,
    .symmetric_threshold = true,
    .red = {.x = (float)160 / 1280, .y = (float)652 / 720, .width = (float)360 / 1280, .height = (float)16 / 720 },
    .green = {.x = (float)720 / 1280, .y = (float)652 / 720, .width = (float)430 / 1280, .height = (float)16 / 720 },
    .red_score = {.x = (float)517 / 1280, .y = (float)612 / 720, .width = (float)40 / 1280, .height = (float)32 / 720 },
    .green_score = {.x = (float)718 / 1280, .y = (float)612 / 720, .width = (float)40 / 1280, .height = (float)32 / 720 },
    .red_name = {.x = (float)128 / 640, .y = (float)305 / 360, .width = (float)140 / 640, .height = (float)20 / 360 },
    .green_name = {.x = (float)374 / 640, .y = (float)305 / 360, .width = (float)140 / 640, .height = (float)20 / 360 },
    .red_country = {.x = (float)212 / 1829, .y = (float)896 / 1031, .width = (float)64 / 1829, .height = (float)24 / 1031 },
    .green_country = {.x = (float)1562 / 1829, .y = (float)896 / 1031, .width = (float)64 / 1829, .height = (float)24 / 1031 },
    .time = {.x = (float)860 / 1829, .y = (float)886 / 1031, .width = (float)108 / 1829, .height = (float)30 / 1031 },
    .tableau = {.x = (float)860 / 1829, .y = (float)938 / 1031, .width = (float)108 / 1829, .height = (float)36 / 1031 },
};

const OverlayConfig OVERLAY_USA_STANDARD = {
    .id = 2,
    .threshold = 0.135,
    .symmetric_threshold = true,
    .red = {.x = (float)0 / 1830, .y = (float)950 / 1030, .width = (float)624 / 1830, .height = (float)75 / 1030 },
    .green = {.x = (float)1200 / 1830, .y = (float)950 / 1030, .width = (float)624 / 1830, .height = (float)75 / 1030 },
    .red_score = {.x = (float)624 / 1830, .y = (float)950 / 1030, .width = (float)100 / 1830, .height = (float)75 / 1030 },
    .green_score = {.x = (float)1100 / 1830, .y = (float)950 / 1030, .width = (float)100 / 1830, .height = (float)75 / 1030 },
    .red_name = {.x = (float)0 / 1830, .y = (float)950 / 1030, .width = (float)624 / 1830, .height = (float)75 / 1030 },
    .green_name = {.x = (float)1200 / 1830, .y = (float)950 / 1030, .width = (float)624 / 1830, .height = (float)75 / 1030 },
    .time = {.x = (float)867 / 1830, .y = (float)950 / 1030, .width = (float)100 / 1830, .height = (float)48 / 1030 },
};

const OverlayConfig OVERLAY_TURKEY = {
    .id = 3,
    .threshold = 0.085,
    .symmetric_threshold = true,
    .red = {.x = (float)1446 / 1835, .y = (float)108 / 1030, .width = (float)50 / 1835, .height = (float)30 / 1030 },
    .green = {.x = (float)1678 / 1835, .y = (float)108 / 1030, .width = (float)50 / 1835, .height = (float)30 / 1030 },
    .red_score = {.x = (float)1475 / 1835, .y = (float)146 / 1030, .width = (float)48 / 1835, .height = (float)32 / 1030 },
    .green_score = {.x = (float)1653 / 1835, .y = (float)146 / 1030, .width = (float)48 / 1835, .height = (float)32 / 1030 },
    .red_name = {.x = (float)1335 / 1835, .y = (float)8 / 1030, .width = (float)250 / 1835, .height = (float)32 / 1030 },
    .green_name = {.x = (float)1585 / 1835, .y = (float)8 / 1030, .width = (float)250 / 1835, .height = (float)32 / 1030 },
    .time = {.x = (float)1538 / 1830, .y = (float)60 / 1030, .width = (float)100 / 1830, .height = (float)40 / 1030 },
};

const OverlayConfig OVERLAYS[] = { OVERLAY_STANDARD_1, OVERLAY_STANDARD_2, OVERLAY_USA_STANDARD, OVERLAY_TURKEY };

void train_nn()
{
    std::cout << "[DIGIT TRAINER] Loading images..." << std::endl;
    // Load your training images and labels
    std::vector<cv::Mat> images; // Your training images (64x64)
    std::vector<int> labels;     // Corresponding labels (digits)

    for (int i = 0; i < 10; ++i) {
        std::string labelFolder = "./foreign/kaggle/" + std::to_string(i);
        for (const auto& entry : std::filesystem::directory_iterator(labelFolder)) {
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            cv::resize(img, img, cv::Size(64, 64));  // Ensure 64x64 size
            images.push_back(img);
            labels.push_back(i);
        }
    }
    std::cout << "[DIGIT TRAINER] Converting images to feature vectors..." << std::endl;

    // Convert images to feature vectors (flatten 64x64 image to 1D)
    std::vector<cv::Mat> trainingData;
    for (const auto& img : images) {
        cv::Mat flatImg = img.reshape(1, 1); // Flatten the image
        trainingData.push_back(flatImg);
    }
    std::cout << "[DIGIT TRAINER] Converting to matrix..." << std::endl;

    cv::Mat trainDataMat;
    cv::vconcat(trainingData, trainDataMat);  // Combine images into a single matrix (vertical concatenation)
    trainDataMat.convertTo(trainDataMat, CV_32F);
    cv::Mat labelsMat(labels);
    labelsMat.convertTo(labelsMat, CV_32SC1);

    // Create SVM
    std::cout << "[DIGIT TRAINER] Creating SVM..." << std::endl;
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setKernel(cv::ml::SVM::LINEAR);   // Use a linear kernel
    svm->setType(cv::ml::SVM::C_SVC);      // Use the C-Support Vector Classification type
    svm->setC(1);                  // Regularization parameter

    // Train the SVM
    std::cout << "[DIGIT TRAINER] Training SVM..." << std::endl;
    svm->train(trainDataMat, cv::ml::ROW_SAMPLE, labelsMat);

    // Save the trained model
    std::cout << "[DIGIT TRAINER] Saving model..." << std::endl;
    svm->save("./foreign/svm_model.xml");
    std::cout << "[DIGIT TRAINER] Training complete!" << std::endl;
}

int classify_digit(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& region)
{
    cv::Mat invertedMat;
    cv::bitwise_not(region, invertedMat);

    // Step 1: Find contours and bounding rectangle
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(invertedMat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int largestIndex = 0;
    double maxArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largestIndex = i;
        }
    }
    cv::Rect boundingRect = cv::boundingRect(contours[largestIndex]);

    // Step 2: Crop the black region
    cv::Mat cropped = region(boundingRect);

    int width = (double)48 * boundingRect.width / boundingRect.height;
    int height = 48;
    // Step 3: Resize the cropped region to 48x48
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(width, height));

    // Step 4: Center the resized region in a 64x64 mat
    cv::Mat centeredMat = cv::Mat::ones(64, 64, region.type()) * 255;
    int x_offset = (64 - width) / 2;
    int y_offset = (64 - height) / 2;

    if (x_offset < 0)
    {
        width -= x_offset;
        x_offset = 0;
    }
    if (y_offset < 0)
    {
        height -= boundingRect.y;
        y_offset = 0;
    }
    if (width > 64) width = 64;
    if (height > 64) height = 64;

    try {
        resized.copyTo(centeredMat(cv::Rect(x_offset, y_offset, width, height)));
    }
    catch (const std::exception& e) {
        std::cout << "Trying to resize with rect " << cv::Rect(x_offset, y_offset, width, height) << std::endl;
        std::cerr << "Error in digit classification: " << e.what() << std::endl;
        return -1;
    }

    // Flatten the image (reshape from 64x64 to 1D vector)
    cv::Mat flatImg = centeredMat.reshape(1, 1); // Flatten to 1D vector

    // Convert the image to CV_32F (float) as the SVM expects this type
    flatImg.convertTo(flatImg, CV_32F);


    // Use the SVM to predict the digit
    float predictedLabel = svm->predict(flatImg);

    return predictedLabel;
}

int8_t classify_score(const cv::Ptr<cv::ml::SVM>& svm, cv::Mat& region, OverlayConfig overlay)
{
    // Convert image to gray
    cv::Mat gray;
    cv::cvtColor(region, gray, cv::COLOR_RGB2GRAY);

    // Threshold to make gray values white
    cv::Mat thresholded;
    if (overlay.id == 0)
    {
        cv::threshold(gray, thresholded, 225, 255, cv::THRESH_BINARY_INV);
    }
    else if (overlay.id == 1)
    {
        cv::threshold(gray, thresholded, 110, 255, cv::THRESH_BINARY);
    }
    else if (overlay.id == 2)
    {
        cv::threshold(gray, thresholded, 110, 255, cv::THRESH_BINARY_INV);
    }
    else if (overlay.id == 3)
    {
        cv::threshold(gray, thresholded, 135, 255, cv::THRESH_BINARY_INV);
    }

    cv::Mat left_mat = thresholded(cv::Rect(0, 0, thresholded.cols / 2, thresholded.rows));
    cv::Mat right_mat = thresholded(cv::Rect(thresholded.cols / 2, 0, thresholded.cols / 2, thresholded.rows));

    bool over_10 = false;

    if (overlay.symmetric_threshold)
    {
        if (overlay.id == 3 || true) {
            std::vector<int> colSums(thresholded.cols, 0);

            for (int x = 0; x < thresholded.cols; x++) { // Compute column-wise black pixel count
                colSums[x] = thresholded.rows - cv::countNonZero(thresholded.col(x)); // Count black pixels
            }

            double meanX = 0, totalWeight = 0; // Compute mean column position (center of mass)
            for (int x = 0; x < thresholded.cols; x++) {
                meanX += x * colSums[x];
                totalWeight += colSums[x];
            }
            meanX /= (totalWeight + 1e-5); // Avoid division by zero

            double spread = 0; // Compute spread: sum of absolute deviations from center
            for (int x = 0; x < thresholded.cols; x++) {
                spread += std::abs(x - meanX) * colSums[x];
            }
            spread /= (totalWeight + 1e-5);

            over_10 = spread > 0.1 * thresholded.cols;
        }
        else
        {
            int left_content = (left_mat.rows * left_mat.cols) - cv::countNonZero(left_mat);
            int right_content = (right_mat.rows * right_mat.cols) - cv::countNonZero(right_mat);
            int threshold = (overlay.threshold * thresholded.rows * thresholded.cols);
            over_10 = left_content > threshold && right_content > threshold;
            // std::cout << "Left: " << left_content << ", Right: " << right_content << ", Threshold: " << threshold << std::endl;
        }
    }
    else
    {
        int contentCount = (thresholded.rows * thresholded.cols) - cv::countNonZero(thresholded);
        over_10 = contentCount > (overlay.threshold * thresholded.rows * thresholded.cols);
        // std::cout << "Content count: " << contentCount << ", threshold: " << (overlay.threshold * thresholded.rows * thresholded.cols) << std::endl;
    }

    // imshow("tracker", thresholded);
    // cv::waitKey(500);

    if (over_10)
    {
        int left_digit = classify_digit(svm, left_mat);
        int right_digit = classify_digit(svm, right_mat);
        if (left_digit == -1 || right_digit == -1) return -1;
        return left_digit * 10 + right_digit;
    }
    else
    {
        return classify_digit(svm, thresholded);
    }
}

cv::Rect roiFromVideoInfo(int width, int height, VideoROI roi)
{
    return cv::Rect(width * roi.x, height * roi.y, width * roi.width, height * roi.height);
}

cv::Ptr<cv::ml::SVM> preload_digit_model(const std::string& svm_path)
{
    std::filesystem::path exe_path = std::filesystem::current_path();
    std::cout << "[DIGIT MODEL] Looking for svm model at " << svm_path << std::endl;
    if (!std::filesystem::exists(svm_path))
    {
        train_nn();
    }
    else
    {
        std::cout << "[DIGIT MODEL] svm_model.xml exists, skipping" << std::endl;
    }
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svm_path);
    std::cout << "[DIGIT MODEL] Model loaded successfully!" << std::endl;
    return svm;
}

void js_memcpy(void* dest, void* source, size_t size) {
    memcpy(dest, source, size);
}

std::string formatString(const std::string& input, bool numeric_allowed, bool uppercase = false) {
    size_t start = 0, end = input.size();

    // Trim from both ends
    while (start < end && !(numeric_allowed ? std::isalnum(input[start]) : std::isalpha(input[start]))) ++start;
    while (end > start && !(numeric_allowed ? std::isalnum(input[end - 1]) : std::isalpha(input[end - 1]))) --end;

    if (start >= end) return "";

    std::string result;
    result.reserve(end - start);

    bool capitalize = true;
    for (size_t i = start; i < end; ++i) {
        if (std::isspace(input[i])) {
            if (!result.empty() && result.back() != ' ') result.push_back(' ');
            capitalize = true;
        }
        else {
            if (uppercase) {
                result.push_back(std::toupper(input[i]));
            }
            else {
                result.push_back(capitalize ? std::toupper(input[i]) : std::tolower(input[i]));
            }
            capitalize = false;
        }
    }

    return result;
}

#pragma warning(disable: 4576)

static void log_packet(const AVFormatContext* fmt_ctx, const AVPacket* pkt, const char* tag)
{
    AVRational* time_base = &fmt_ctx->streams[pkt->stream_index]->time_base;

    printf("%s: pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s stream_index:%d\n",
        tag,
        av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, time_base),
        av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, time_base),
        av_ts2str(pkt->duration), av_ts2timestr(pkt->duration, time_base),
        pkt->stream_index);
}


/**
 * @brief Print the information of the passed packet.
 *
 * @fn logPacket
 * @param avFormatContext AVFormatContext of the given packet.
 * @param avPacket AVPacket to log.
 * @param tag String to tag the log output.
 */
void logPacket(const AVFormatContext* avFormatContext, const AVPacket* avPacket, const std::string& tag) {
    return;
    AVRational* timeBase = &avFormatContext->streams[avPacket->stream_index]->time_base;

    std::cout << tag << ": pts:" << av_ts2str(avPacket->pts)
        << " pts_time:" << av_ts2timestr(avPacket->pts, timeBase)
        << " dts:" << av_ts2str(avPacket->dts)
        << " dts_time:" << av_ts2timestr(avPacket->dts, timeBase)
        << " duration:" << av_ts2str(avPacket->duration)
        << " duration_time:" << av_ts2timestr(avPacket->duration, timeBase)
        << " stream_index:" << avPacket->stream_index << std::endl;
}

/**
 * @brief Cut a file in the given input file path based on the start and end seconds, and output the cutted file to the
 * given output file path.
 *
 * @fn cut_file
 * @param inputFilePath Input file path to be cutted.
 * @param startSeconds Cutting start time in seconds.
 * @param endSeconds Cutting end time in seconds.
 * @param outputFilePath Output file path to write the new cutted file.
 *
 * @details This function will take an input file path and cut it based on the given start and end seconds. The cutted
 * file will then be outputted to the given output file path.
 *
 * @return True if the cutting operation finished successfully, false otherwise.
 */
bool cut_file(const std::string& inputFilePath, const long long& startSeconds, const long long& endSeconds,
    const std::string& outputFilePath, const int framesToSkip) {
    int operationResult;

    AVPacket* avPacket = NULL;
    AVFormatContext* avInputFormatContext = NULL;
    AVFormatContext* avOutputFormatContext = NULL;

    avPacket = av_packet_alloc();
    if (!avPacket) {
        std::cerr << "Failed to allocate AVPacket." << std::endl;
        return false;
    }

    try {
        operationResult = avformat_open_input(&avInputFormatContext, inputFilePath.c_str(), 0, 0);
        if (operationResult < 0) {
            throw std::runtime_error("Failed to open the input file.");
        }

        operationResult = avformat_find_stream_info(avInputFormatContext, 0);
        if (operationResult < 0) {
            throw std::runtime_error("Failed to retrieve the input stream information.");
        }

        avformat_alloc_output_context2(&avOutputFormatContext, NULL, NULL, outputFilePath.c_str());
        if (!avOutputFormatContext) {
            operationResult = AVERROR_UNKNOWN;
            throw std::runtime_error("Failed to create the output context.");
        }

        int streamIndex = 0;
        int* streamMapping = new int[avInputFormatContext->nb_streams];
        int* streamRescaledStartSeconds = new int[avInputFormatContext->nb_streams];
        int* streamRescaledEndSeconds = new int[avInputFormatContext->nb_streams];

        // Copy streams from the input file to the output file.
        for (int i = 0; i < avInputFormatContext->nb_streams; i++) {
            AVStream* outStream;
            AVStream* inStream = avInputFormatContext->streams[i];

            streamRescaledStartSeconds[i] = av_rescale_q((startSeconds + framesToSkip) * AV_TIME_BASE, AV_TIME_BASE_Q, inStream->time_base);
            streamRescaledEndSeconds[i] = av_rescale_q(endSeconds * AV_TIME_BASE, AV_TIME_BASE_Q, inStream->time_base);

            if (inStream->codecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
                inStream->codecpar->codec_type != AVMEDIA_TYPE_VIDEO &&
                inStream->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE) {
                streamMapping[i] = -1;
                continue;
            }

            streamMapping[i] = streamIndex++;

            outStream = avformat_new_stream(avOutputFormatContext, NULL);
            if (!outStream) {
                operationResult = AVERROR_UNKNOWN;
                throw std::runtime_error("Failed to allocate the output stream.");
            }

            operationResult = avcodec_parameters_copy(outStream->codecpar, inStream->codecpar);
            if (operationResult < 0) {
                throw std::runtime_error("Failed to copy codec parameters from input stream to output stream.");
            }
            outStream->codecpar->codec_tag = 0;
        }

        if (!(avOutputFormatContext->oformat->flags & AVFMT_NOFILE)) {
            operationResult = avio_open(&avOutputFormatContext->pb, outputFilePath.c_str(), AVIO_FLAG_WRITE);
            if (operationResult < 0) {
                throw std::runtime_error("Failed to open the output file.");
            }
        }

        operationResult = avformat_write_header(avOutputFormatContext, NULL);
        if (operationResult < 0) {
            throw std::runtime_error("Error occurred when opening output file.");
        }

        // operationResult = avformat_seek_file(avInputFormatContext, -1, INT64_MIN, startSeconds * AV_TIME_BASE,
        //     startSeconds * AV_TIME_BASE, 0);
        av_seek_frame(avInputFormatContext, -1, startSeconds * AV_TIME_BASE, AVSEEK_FLAG_BACKWARD);

        if (operationResult < 0) {
            throw std::runtime_error("Failed to seek the input file to the targeted start position.");
        }

        bool foundKeyframe = false;
        while (true) {
            operationResult = av_read_frame(avInputFormatContext, avPacket);
            if (operationResult < 0) break;

            // Skip packets from unknown streams and packets after the end cut position.
            if (avPacket->stream_index >= avInputFormatContext->nb_streams || streamMapping[avPacket->stream_index] < 0 ||
                avPacket->pts > streamRescaledEndSeconds[avPacket->stream_index]) {
                av_packet_unref(avPacket);
                continue;
            }

            avPacket->stream_index = streamMapping[avPacket->stream_index];
            logPacket(avInputFormatContext, avPacket, "in");

            // Shift the packet to its new position by subtracting the rescaled start seconds.
            avPacket->pts -= streamRescaledStartSeconds[avPacket->stream_index];
            avPacket->dts -= streamRescaledStartSeconds[avPacket->stream_index];

            av_packet_rescale_ts(avPacket, avInputFormatContext->streams[avPacket->stream_index]->time_base,
                avOutputFormatContext->streams[avPacket->stream_index]->time_base);
            avPacket->pos = -1;
            logPacket(avOutputFormatContext, avPacket, "out");

            operationResult = av_interleaved_write_frame(avOutputFormatContext, avPacket);
            if (operationResult < 0) {
                throw std::runtime_error("Failed to mux the packet.");
            }
        }

        delete[] streamMapping;
        delete[] streamRescaledStartSeconds;
        delete[] streamRescaledEndSeconds;

        av_write_trailer(avOutputFormatContext);
    }
    catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    av_packet_free(&avPacket);

    avformat_close_input(&avInputFormatContext);

    if (avOutputFormatContext && !(avOutputFormatContext->oformat->flags & AVFMT_NOFILE))
        avio_closep(&avOutputFormatContext->pb);
    avformat_free_context(avOutputFormatContext);

    if (operationResult < 0 && operationResult != AVERROR_EOF) {
        char buffer[64] = {};
        std::cerr << "Error occurred: " << av_make_error_string(buffer, 64, operationResult) << std::endl;
        return false;
    }

    return true;
}

extern "C" StreamAnalysis* cut_stream(const std::string& tesseract_path, const std::string& svm_path, const std::string& video_path, uint8_t overlay_id, const std::string& output_folder, const std::string& event_name, void (*callback)(int))
{
    std::cout << "Tesseract path: " << tesseract_path <<
        "\nSVM path: " << svm_path <<
        "\nVideo path: " << video_path <<
        "\nOverlay ID: " << (int)overlay_id <<
        "\nOutput folder: " << output_folder << std::endl;

    StreamAnalysis* analysis = new StreamAnalysis();

    cv::Ptr<cv::ml::SVM> svm = preload_digit_model(svm_path);

    tesseract::TessBaseAPI tess;
    if (tess.Init(tesseract_path.c_str(), "eng", tesseract::OEM_LSTM_ONLY)) // Initialize Tesseract with English language
    {
        std::cerr << "Could not initialize Tesseract." << std::endl;
        return analysis;
    }

    // How many seconds to skip at a time while waiting for a bout
    const int SKIP_SECONDS = 15;
    // Minimum length of bout (to prevent fluctuations in score)
    const int MIN_BOUT_SECONDS = 60;

    tess.SetVariable("user_words_suffix", COUNTRY_CODES);

    try
    {
        cv::VideoCapture cap(video_path, cv::CAP_FFMPEG);
        cap.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
        if (!cap.isOpened())
        {
            std::cout << "Failed to open video from " << video_path << std::endl;
            return analysis;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
        OverlayConfig overlay = OVERLAYS[overlay_id];

        cv::Mat frame;

        if (overlay_id == 3)
        {
            cap.read(frame);
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            while (true) {
                cv::Mat drawingFrame = frame.clone();

                cv::Rect overlayRect = cv::selectROI("Select the overlay region", drawingFrame);
                int x = overlayRect.x;
                int y = 0;
                int width = drawingFrame.cols - overlayRect.x - 1;
                int height = overlayRect.y + overlayRect.height;

                cv::Rect redNameRect(overlayRect.x, height * 0.02, width / 2, height * 0.2);
                cv::Rect greenNameRect(overlayRect.x + width / 2, height * 0.02, width / 2, height * 0.2);
                cv::rectangle(drawingFrame, redNameRect, cv::Scalar(0, 0, 255), 2);
                cv::rectangle(drawingFrame, greenNameRect, cv::Scalar(0, 255, 0), 2);

                cv::Rect timeRect(overlayRect.x + width * 0.4, height * 0.33, width * 0.2, height * 0.2);
                cv::rectangle(drawingFrame, timeRect, cv::Scalar(255, 0, 0), 2);

                cv::Rect redScoreRect(overlayRect.x + width * 0.275, height * 0.78, width * 0.1, height * 0.18);
                cv::Rect greenScoreRect(overlayRect.x + width * 0.635, height * 0.78, width * 0.1, height * 0.18);
                cv::rectangle(drawingFrame, redScoreRect, cv::Scalar(0, 0, 255), 2);
                cv::rectangle(drawingFrame, greenScoreRect, cv::Scalar(0, 255, 0), 2);
                imshow("Press BACKSPACE if the overlay does not line up, otherwise press ENTER", drawingFrame);
                int key = cv::waitKey(5000);
                std::cout << "Got key: " << key << std::endl;
                if (key == 127 || key == 8) // Backspace
                {
                    cv::destroyWindow("Select the overlay region");
                    cv::destroyWindow("Press BACKSPACE if the overlay does not line up, otherwise press ENTER");
                    continue;
                }

                overlay.red_name.fromRect(redNameRect, frame.cols, frame.rows);
                overlay.green_name.fromRect(greenNameRect, frame.cols, frame.rows);
                overlay.time.fromRect(timeRect, frame.cols, frame.rows);
                overlay.red_score.fromRect(redScoreRect, frame.cols, frame.rows);
                overlay.green_score.fromRect(greenScoreRect, frame.cols, frame.rows);

                cv::destroyWindow("Select the overlay region");
                cv::destroyWindow("Press BACKSPACE if the overlay does not line up, otherwise press ENTER");
                break;
            }
        }

        analysis->fps = fps;
        analysis->frame_count = frame_count;

        std::cout << "Selected overlay: " << overlay << "\n"
            << "FPS: " << fps << "\n"
            << "Width: " << width << "\n"
            << "Height: " << height << "\n"
            << "Total Frames: " << frame_count << std::endl;

        cv::Rect redRoi = roiFromVideoInfo(width, height, overlay.red);
        cv::Rect greenRoi = roiFromVideoInfo(width, height, overlay.green);
        cv::Rect redScoreRoi = roiFromVideoInfo(width, height, overlay.red_score);
        cv::Rect greenScoreRoi = roiFromVideoInfo(width, height, overlay.green_score);
        cv::Rect redNameRoi = roiFromVideoInfo(width, height, overlay.red_name);
        cv::Rect greenNameRoi = roiFromVideoInfo(width, height, overlay.green_name);
        cv::Rect redCountryRoi = roiFromVideoInfo(width, height, overlay.red_country);
        cv::Rect greenCountryRoi = roiFromVideoInfo(width, height, overlay.green_country);
        cv::Rect timeRoi = roiFromVideoInfo(width, height, overlay.time);
        cv::Rect tableauRoi = roiFromVideoInfo(width, height, overlay.tableau);

        int skip_rate = SKIP_SECONDS * fps;
        int min_bout_length = MIN_BOUT_SECONDS * fps;

        int frames_to_skip = 0;

        bool bout_running = false;
        callback(1);
        for (int i = 0; i < frame_count; i++)
        {
            if (frames_to_skip > 0)
            {
                cap.set(cv::CAP_PROP_POS_FRAMES, i + frames_to_skip);
                i += frames_to_skip;
                frames_to_skip = 0;
                continue;
            }
            cap.read(frame);

            cv::Mat redScoreMask = frame(redScoreRoi);
            cv::Mat greenScoreMask = frame(greenScoreRoi);
            int8_t redScore = classify_score(svm, redScoreMask, overlay);
            int8_t greenScore = classify_score(svm, greenScoreMask, overlay);
            if (redScore < 0 || greenScore < 0)
            {
                continue;
            }
            bool score_nonzero = redScore > 0 || greenScore > 0;

            std::cout << "Score at frame " << i << " (" << i / fps << "s, " << i * 100 / frame_count << "%): " << (int)redScore << "-" << (int)greenScore << std::endl;

            if (!bout_running && score_nonzero)
            {
                tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
                callback(i * 100 / frame_count);
                bout_running = true;
                analysis->bouts[analysis->bout_count].start_frame = i - skip_rate * 4;

                for (int rewind_amount = 0; rewind_amount < 100; rewind_amount++)
                {
                    cv::Mat local_frame;
                    cap.set(cv::CAP_PROP_POS_FRAMES, i - rewind_amount * skip_rate);
                    cap.read(local_frame);

                    cv::Mat redScoreMask = frame(redScoreRoi);
                    cv::Mat greenScoreMask = frame(greenScoreRoi);
                    int8_t redScore = classify_score(svm, redScoreMask, overlay);
                    int8_t greenScore = classify_score(svm, greenScoreMask, overlay);
                    if (redScore > 0 || greenScore > 0) continue;

                    cv::Mat timeMat = local_frame(timeRoi);
                    cv::cvtColor(timeMat, timeMat, cv::COLOR_BGR2GRAY);
                    tess.SetImage(timeMat.data, timeMat.cols, timeMat.rows, 1, timeMat.step); // Feed binary image to Tesseract
                    std::string time_str = tess.GetUTF8Text();
                    if (time_str.length() == 0 || time_str[0] == '3')
                    {
                        analysis->bouts[analysis->bout_count].start_frame = i - rewind_amount * skip_rate;
                        break;
                    }
                }
                cap.set(cv::CAP_PROP_POS_FRAMES, i);
                std::cout << "Bout started at frame " << i << " (" << i / fps << "s, " << i * 100 / frame_count << "%)!" << std::endl;

                {
                    cv::Mat redName = frame(redNameRoi);
                    cv::cvtColor(redName, redName, cv::COLOR_BGR2GRAY);
                    tess.SetImage(redName.data, redName.cols, redName.rows, 1, redName.step);
                    std::string red_str = tess.GetUTF8Text();
                    red_str = formatString(red_str, false);

                    cv::Mat greenName = frame(greenNameRoi);
                    cv::cvtColor(greenName, greenName, cv::COLOR_BGR2GRAY);
                    tess.SetImage(greenName.data, greenName.cols, greenName.rows, 1, greenName.step);
                    std::string green_str = tess.GetUTF8Text();
                    green_str = formatString(green_str, false);

                    if (tableauRoi.area() > 1)
                    {
                        cv::Mat tableau = frame(tableauRoi);
                        cv::cvtColor(tableau, tableau, cv::COLOR_BGR2GRAY);
                        tess.SetImage(tableau.data, tableau.cols, tableau.rows, 1, tableau.step);
                        std::string tableau_str = tess.GetUTF8Text();
                        tableau_str = formatString(tableau_str, true);
                        analysis->bouts[analysis->bout_count].tableau = strdup(tableau_str.c_str());
                        std::cout << "Tableau: " << tableau_str << ", ";
                    }
                    else
                    {
                        analysis->bouts[analysis->bout_count].tableau = nullptr;
                    }

                    tess.SetVariable("lang_model_penalty_non_dict_word", "0");
                    tess.SetPageSegMode(tesseract::PSM_SINGLE_WORD);

                    std::cout << "Red name: " << red_str;

                    if (redCountryRoi.area() > 1)
                    {
                        cv::Mat redCountry = frame(redCountryRoi);
                        cv::cvtColor(redCountry, redCountry, cv::COLOR_BGR2GRAY);
                        tess.SetImage(redCountry.data, redCountry.cols, redCountry.rows, 1, redCountry.step);
                        std::string red_country = tess.GetUTF8Text();
                        red_country = formatString(red_country, false, true);
                        analysis->bouts[analysis->bout_count].country_left = red_country.length() > 0 ? strdup(red_country.c_str()) : nullptr;
                        std::cout << " (" << red_country << "),";
                    }
                    else
                    {
                        analysis->bouts[analysis->bout_count].country_left = nullptr;
                    }

                    std::cout << "Green name: " << green_str;

                    if (greenCountryRoi.area() > 1)
                    {
                        cv::Mat greenCountry = frame(greenCountryRoi);
                        cv::cvtColor(greenCountry, greenCountry, cv::COLOR_BGR2GRAY);
                        tess.SetImage(greenCountry.data, greenCountry.cols, greenCountry.rows, 1, greenCountry.step);
                        std::string green_country = tess.GetUTF8Text();
                        green_country = formatString(green_country, false, true);
                        analysis->bouts[analysis->bout_count].country_right = green_country.length() > 0 ? strdup(green_country.c_str()) : nullptr;
                        std::cout << " (" << green_country << ")";
                    }
                    else
                    {
                        analysis->bouts[analysis->bout_count].country_right = nullptr;
                    }

                    std::cout << std::endl;

                    analysis->bouts[analysis->bout_count].name_left = strdup(red_str.c_str());
                    analysis->bouts[analysis->bout_count].name_right = strdup(green_str.c_str());

                    tess.SetVariable("lang_model_penalty_non_dict_word", "0.5");
                }
            }
            if (bout_running && !score_nonzero)
            {
                // cv::Mat local_frame;
                // cap.set(cv::CAP_PROP_POS_FRAMES, i + skip_rate);
                // cap.read(local_frame);

                // cv::Mat redScoreMask = local_frame(redScoreRoi);
                // cv::Mat greenScoreMask = local_frame(greenScoreRoi);
                // int8_t redScore = classify_score(svm, redScoreMask, overlay);
                // int8_t greenScore = classify_score(svm, greenScoreMask, overlay);
                // if (redScore > 0 || greenScore > 0)
                // {
                //     std::cout << "Score hit 0 at frame " << i << ", but score restored immediately after" << std::endl;
                // }
                // else
                // {
                std::cout << "Bout ended at frame " << i << " (" << i / fps << "s, " << i * 100 / frame_count << "%)!" << std::endl;
                bout_running = false;
                if (i - analysis->bouts[analysis->bout_count].start_frame > min_bout_length)
                {
                    analysis->bouts[analysis->bout_count].end_frame = i;
                    analysis->bout_count += 1;
                }
                // }
            }
            // cap.set(cv::CAP_PROP_POS_FRAMES, i);

            frames_to_skip = skip_rate;
        }
        if (bout_running)
        {
            analysis->bouts[analysis->bout_count].end_frame = frame_count - 1;
            analysis->bout_count += 1;
        }

        callback(-1);
        for (int i = 0; i < analysis->bout_count; i++)
        {
            StreamBoutSegment bout = analysis->bouts[i];
            double start = bout.start_frame / fps;
            double end = bout.end_frame / fps;
            std::string fencer_name_section = std::string("/");
            if (event_name.length() > 0) fencer_name_section += event_name + std::string(" ");
            fencer_name_section += bout.name_left;
            if (bout.country_left != nullptr) fencer_name_section += std::string(" ") + bout.country_left;
            fencer_name_section += std::string(" vs ") + bout.name_right;
            if (bout.country_right != nullptr) fencer_name_section += std::string(" ") + bout.country_right;
            if (bout.tableau != nullptr) fencer_name_section += std::string(" ") + bout.tableau;
            std::string bout_name = fencer_name_section + ".mp4";
            std::cout << "Start: " << start << ", End: " << end << ", Duration: " << end - start << std::endl;
            const char* video_name = (std::string(output_folder) + bout_name).c_str();
            std::cout << "Input: " << video_path << " Output: " << video_name << std::endl;
            cut_file(video_path, start, end, std::string(output_folder) + bout_name, 0);
            callback(-(i + 1) * 100 / analysis->bout_count);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return analysis;
}

#include <windows.h>

void cut_stream_async(const char* tesseract_path, const char* svm_path, const char* video_path, uint8_t overlay_id, const char* output_folder, const char* event_name, void (*callback)(int)) {
    std::string tesseract_str(tesseract_path);
    std::string svm_str(svm_path);
    std::string video_path_str(video_path);
    std::string output_folder_str(output_folder);
    std::string event_name_str(event_name);

    std::thread([tesseract_str, svm_str, video_path_str, overlay_id, output_folder_str, event_name_str, callback]() {
        try {

            cut_stream(tesseract_str, svm_str, video_path_str, overlay_id, output_folder_str, event_name_str, callback);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
        }

        // Call callback with result
        if (callback) callback(255);
        }).detach(); // Detach thread to run independently
}