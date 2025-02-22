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
    EXPORT void cut_stream_async(const char* tesseract_path, const char* svm_path, const char* video_path, uint8_t overlay_id, const char* output_folder, void (*callback)(int));
}

int32_t add(int32_t a, int32_t b) {
    return a + b;
}

struct VideoROI {
    float x;
    float y;
    float width;
    float height;
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

const OverlayConfig OVERLAYS[] = { OVERLAY_STANDARD_1, OVERLAY_STANDARD_2 };

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

    resized.copyTo(centeredMat(cv::Rect(x_offset, y_offset, width, height)));

    // Flatten the image (reshape from 64x64 to 1D vector)
    cv::Mat flatImg = centeredMat.reshape(1, 1); // Flatten to 1D vector

    // Convert the image to CV_32F (float) as the SVM expects this type
    flatImg.convertTo(flatImg, CV_32F);

    // Use the SVM to predict the digit
    float predictedLabel = svm->predict(flatImg);

    return predictedLabel;
}

uint8_t classify_score(const cv::Ptr<cv::ml::SVM>& svm, cv::Mat& region, OverlayConfig overlay)
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

    cv::Mat left_mat = thresholded(cv::Rect(0, 0, thresholded.cols / 2, thresholded.rows));
    cv::Mat right_mat = thresholded(cv::Rect(thresholded.cols / 2, 0, thresholded.cols / 2, thresholded.rows));

    bool over_10 = false;

    if (overlay.symmetric_threshold)
    {
        int left_content = (left_mat.rows * left_mat.cols) - cv::countNonZero(left_mat);
        int right_content = (right_mat.rows * right_mat.cols) - cv::countNonZero(right_mat);
        over_10 = left_content > 10 && right_content > 10;
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
        return 10 + classify_digit(svm, right_mat);
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

VideoAnalysis* process(const char* video_path, OverlayConfig overlay)
{
    VideoAnalysis* analysis = new VideoAnalysis();

    cv::Ptr<cv::ml::SVM> svm = preload_digit_model("./foreign/svm_model.xml");

    int SKIP_RATE = 20;
    int SCORE_CHECK_DELAY = 450;

    cv::VideoCapture cap(video_path, cv::CAP_FFMPEG);
    cap.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
    if (!cap.isOpened()) return 0;

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::cout << "Selected overlay: " << overlay << std::endl;

    std::cout << "FPS: " << fps << "\n"
        << "Width: " << width << "\n"
        << "Height: " << height << "\n"
        << "Total Frames: " << frame_count << std::endl;

    cv::Rect redRoi = roiFromVideoInfo(width, height, overlay.red);
    cv::Rect greenRoi = roiFromVideoInfo(width, height, overlay.green);
    cv::Rect redScoreRoi = roiFromVideoInfo(width, height, overlay.red_score);
    cv::Rect greenScoreRoi = roiFromVideoInfo(width, height, overlay.green_score);
    cv::Mat frame;
    cv::Ptr<cv::Tracker> tracker = cv::TrackerMIL::create();

    cap >> frame;

    while (redRoi.width == 0 || redRoi.height == 0)
    {
        // cv::rectangle(frame, redRoi, cv::Scalar(0, 255, 0), 2);
        imshow("tracker", frame);
        redRoi = selectROI("tracker", frame);
        std::cout << "Red ROI: " << redRoi << std::endl;
    }
    while (greenRoi.width == 0 || greenRoi.height == 0)
    {
        greenRoi = selectROI("tracker", frame);
        std::cout << "Green ROI: " << greenRoi << std::endl;
    }
    while (redScoreRoi.width == 0 || redScoreRoi.height == 0)
    {
        redScoreRoi = selectROI("tracker", frame);
        std::cout << "Red score ROI: " << redScoreRoi << std::endl;
    }
    while (greenScoreRoi.width == 0 || greenScoreRoi.height == 0)
    {
        greenScoreRoi = selectROI("tracker", frame);
        std::cout << "Green score ROI: " << greenScoreRoi << std::endl;
    }

    bool redOn = false;
    bool greenOn = false;

    int doubleTimeout = 0;
    int frames_to_skip = SKIP_RATE;
    bool should_check_score = false; // Should we check the score?
    uint32_t delay_check_frame = -1; // Another frame far in the future just in case there was video on 14-14

    uint8_t last_red = -1;
    uint8_t last_green = -1;

    for (int i = 1; i < frame_count; i++) {
        int seconds = i / 30;
        cap >> frame;
        if (should_check_score || i == delay_check_frame)
        {
            if (i == delay_check_frame) std::cout << "Delay check at " << delay_check_frame << std::endl;
            cv::Mat redScoreMask = frame(redScoreRoi);
            cv::Mat greenScoreMask = frame(greenScoreRoi);

            uint8_t redScore = classify_score(svm, redScoreMask, overlay);
            uint8_t greenScore = classify_score(svm, greenScoreMask, overlay);
            std::cout << "[ANALYSIS] Score at frame " << i << ": " << static_cast<int>(redScore) << "-" << static_cast<int>(greenScore) << std::endl;
            // If the score is the same as before, the last touch must have been simul or annuled, so we overwrite
            // if (redScore == last_red && greenScore == last_green)
            // {
            //     analysis->touch_count -= 1;
            //     // If i is the delay check frame, it's not the actual time when it happened
            //     if (should_check_score)
            //     {
            //         analysis->touches[analysis->touch_count].frame = i;
            //     }
            // }
            // else
            // {
            analysis->touches[analysis->touch_count].frame = i;
            // }
            // Update last score accordingly
            if (analysis->touch_count > 0)
            {
                analysis->touches[analysis->touch_count - 1].score1 = redScore;
                analysis->touches[analysis->touch_count - 1].score2 = greenScore;
            }
            analysis->touch_count++;
            last_red = redScore;
            last_green = greenScore;
            should_check_score = false;
        }
        if (frames_to_skip > 0)
        {
            frames_to_skip--;
            continue;
        }
        cv::Mat hsvFrame;
        cv::cvtColor(frame, hsvFrame, cv::COLOR_RGB2HSV);
        cv::Mat redMat = hsvFrame(redRoi);
        cv::Mat greenMat = hsvFrame(greenRoi);

        cv::Mat redMask;
        cv::inRange(redMat, cv::Scalar(120, 100, 160), cv::Scalar(140, 255, 255), redMask);
        int redThreshold = redRoi.area() / 3;
        int redPixels = cv::countNonZero(redMask);

        if (i > 2900 && i < 3000) std::cout << "Red " << redPixels << ", threshold: " << redThreshold << std::endl;

        if (redPixels >= redThreshold && !redOn)
        {
            std::cout << "Red light at " << i << std::endl;
            should_check_score = true;
            // delay_check_frame = MIN(i + SCORE_CHECK_DELAY, frame_count - 3);
            redOn = true;
        }
        if (redPixels < redThreshold / 3 && redOn)
        {
            redOn = false;
            doubleTimeout = i + 50;
        }

        cv::Mat greenMask;
        cv::inRange(greenMat, cv::Scalar(50, 100, 160), cv::Scalar(70, 255, 255), greenMask);
        int greenThreshold = redRoi.area() / 2;
        int greenPixels = cv::countNonZero(greenMask);
        // std::cout << "Green pixels: " << greenPixels << std::endl;
        if (greenPixels > greenThreshold && !greenOn)
        {
            // std::cout << "Green light at " << i << std::endl;
            should_check_score = true;
            // delay_check_frame = MIN(i + SCORE_CHECK_DELAY, frame_count - 3);
            greenOn = true;
        }
        if (greenPixels < greenThreshold / 2 && greenOn)
        {
            greenOn = false;
            doubleTimeout = i + 50;
        }

        if (!redOn && !greenOn)
        {
            frames_to_skip = SKIP_RATE;
        }
    }

    if (analysis->touch_count > 0) analysis->touch_count--;

    return analysis;
}

VideoAnalysis* find_video_touches(const char* video_path, uint8_t overlay_id)
{
    try
    {
        auto result = process(video_path, OVERLAYS[overlay_id]);
        return result;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return 0;
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
        int skippedFrames = 0;
        while (true) {
            operationResult = av_read_frame(avInputFormatContext, avPacket);
            if (operationResult < 0) break;

            // Skip packets from unknown streams and packets after the end cut position.
            if (avPacket->stream_index >= avInputFormatContext->nb_streams || streamMapping[avPacket->stream_index] < 0 ||
                avPacket->pts > streamRescaledEndSeconds[avPacket->stream_index]) {
                av_packet_unref(avPacket);
                continue;
            }

            if (avInputFormatContext->streams[avPacket->stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                if (skippedFrames < framesToSkip) {
                    skippedFrames++;
                    av_packet_unref(avPacket);
                    continue;  // Skip this frame
                }
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

extern "C" StreamAnalysis* cut_stream(const std::string& tesseract_path, const std::string& svm_path, const std::string& video_path, uint8_t overlay_id, const std::string& output_folder, void (*callback)(int))
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

        cv::Mat frame;

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
            uint8_t redScore = classify_score(svm, redScoreMask, overlay);
            uint8_t greenScore = classify_score(svm, greenScoreMask, overlay);
            bool score_nonzero = redScore > 0 || greenScore > 0;

            // std::cout << "Score at frame " << i << " (" << i / fps << "s, "  << i * 100 / frame_count << "%): " << (int)redScore << "-" << (int)greenScore << std::endl;

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

                    cv::Mat redCountry = frame(redCountryRoi);
                    cv::cvtColor(redCountry, redCountry, cv::COLOR_BGR2GRAY);
                    tess.SetImage(redCountry.data, redCountry.cols, redCountry.rows, 1, redCountry.step);
                    std::string red_country = tess.GetUTF8Text();
                    red_country = formatString(red_country, false, true);

                    cv::Mat greenCountry = frame(greenCountryRoi);
                    cv::cvtColor(greenCountry, greenCountry, cv::COLOR_BGR2GRAY);
                    tess.SetImage(greenCountry.data, greenCountry.cols, greenCountry.rows, 1, greenCountry.step);
                    std::string green_country = tess.GetUTF8Text();
                    green_country = formatString(green_country, false, true);

                    std::cout << "Red name: " << red_str << " (" << red_country << "), Green name: " << green_str << " (" << green_country << ")" << std::endl;

                    analysis->bouts[analysis->bout_count].name_left = strdup(red_str.c_str());
                    analysis->bouts[analysis->bout_count].name_right = strdup(green_str.c_str());
                    analysis->bouts[analysis->bout_count].country_left = red_country.length() > 0 ? strdup(red_country.c_str()) : nullptr;
                    analysis->bouts[analysis->bout_count].country_right = green_country.length() > 0 ? strdup(green_country.c_str()) : nullptr;

                    tess.SetVariable("lang_model_penalty_non_dict_word", "0.5");
                }
            }
            if (bout_running && !score_nonzero)
            {
                std::cout << "Bout ended at frame " << i << " (" << i / fps << "s, " << i * 100 / frame_count << "%)!" << std::endl;
                bout_running = false;
                if (i - analysis->bouts[analysis->bout_count].start_frame > min_bout_length)
                {
                    analysis->bouts[analysis->bout_count].end_frame = i;
                    analysis->bout_count += 1;
                }
            }

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
            std::string fencer_name_section = std::string("/") + bout.name_left;
            if (bout.country_left != nullptr) fencer_name_section += std::string(" ") + bout.country_left;
            fencer_name_section += std::string(" vs ") + bout.name_right;
            if (bout.country_right != nullptr) fencer_name_section += std::string(" ") + bout.country_right;
            if (bout.tableau != nullptr) fencer_name_section += std::string(" ") + bout.tableau;
            std::string bout_name = fencer_name_section + ".mp4";
            std::cout << "Start: " << start << ", End: " << end << ", Duration: " << end - start << std::endl;
            const char* video_name = (std::string(output_folder) + bout_name).c_str();
            std::cout << "Input: " << video_path << " Output: " << video_name << std::endl;
            cut_file(video_path, start - 2 * skip_rate, end, std::string(output_folder) + bout_name, 2 * skip_rate);
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

void cut_stream_async(const char* tesseract_path, const char* svm_path, const char* video_path, uint8_t overlay_id, const char* output_folder, void (*callback)(int)) {
    std::string tesseract_str(tesseract_path);
    std::string svm_str(svm_path);
    std::string video_path_str(video_path);
    std::string output_folder_str(output_folder);

    std::thread([tesseract_str, svm_str, video_path_str, overlay_id, output_folder_str, callback]() {
        try {

            cut_stream(tesseract_str, svm_str, video_path_str, overlay_id, output_folder_str, callback);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
        }

        // Call callback with result
        if (callback) callback(255);
        }).detach(); // Detach thread to run independently
}