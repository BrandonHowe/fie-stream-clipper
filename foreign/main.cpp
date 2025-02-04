#include <cstdint>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>

extern "C" int32_t add(int32_t a, int32_t b) {
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
};

std::ostream& operator<<(std::ostream& os, const OverlayConfig& obj) {
    os << "Overlay " << static_cast<int>(obj.id) << ": " << obj.red << ", " << obj.green << ", " << obj.red_score << ", " << obj.green_score;
    return os;
}

const OverlayConfig OVERLAY_STANDARD_1 = {
    .id = 0,
    .threshold = 0.171,
    .symmetric_threshold = false,
    .red = (VideoROI){ .x = (float)228 / 1920, .y = (float)980 / 1080, .width = (float)608 / 1920, .height = (float)32 / 1080 },
    .green = (VideoROI){ .x = (float)1063 / 1920, .y = (float)980 / 1080, .width = (float)608 / 1920, .height = (float)32 / 1080 },
    .red_score = (VideoROI){ .x = (float)792 / 1920, .y = (float)927 / 1080, .width = (float)64 / 1920, .height = (float)48 / 1080 },
    .green_score = (VideoROI){ .x = (float)1060 / 1920, .y = (float)927 / 1080, .width = (float)64 / 1920, .height = (float)48 / 1080 }
};

const OverlayConfig OVERLAY_STANDARD_2 = {
    .id = 1,
    .threshold = 0.19,
    .symmetric_threshold = true,
    .red = (VideoROI){ .x = (float)160 / 1280, .y = (float)652 / 720, .width = (float)360 / 1280, .height = (float)16 / 720 },
    .green = (VideoROI){ .x = (float)720 / 1280, .y = (float)652 / 720, .width = (float)430 / 1280, .height = (float)16 / 720 },
    .red_score = (VideoROI){ .x = (float)517 / 1280, .y = (float)612 / 720, .width = (float)40 / 1280, .height = (float)32 / 720 },
    .green_score = (VideoROI){ .x = (float)718 / 1280, .y = (float)612 / 720, .width = (float)40 / 1280, .height = (float)32 / 720 },
    .red_name = (VideoROI){ .x = (float)128 / 640, .y = (float)305 / 360, .width = (float)140 / 640, .height = (float)20 / 360 },
    .green_name = (VideoROI){ .x = (float)374 / 640, .y = (float)305 / 360, .width = (float)140 / 640, .height = (float)20 / 360 }
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
        std::cout << "Content count: " << contentCount << ", threshold: " << (overlay.threshold * thresholded.rows * thresholded.cols) << std::endl;
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

cv::Ptr<cv::ml::SVM> preload_digit_model(const char* svm_path)
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
        redRoi = selectROI("tracker",frame);
        std::cout << "Red ROI: " << redRoi << std::endl;
    }
    while (greenRoi.width == 0 || greenRoi.height == 0)
    {
        greenRoi = selectROI("tracker",frame);
        std::cout << "Green ROI: " << greenRoi << std::endl;
    }
    while (redScoreRoi.width == 0 || redScoreRoi.height == 0)
    {
        redScoreRoi = selectROI("tracker",frame);
        std::cout << "Red score ROI: " << redScoreRoi << std::endl;
    }
    while (greenScoreRoi.width == 0 || greenScoreRoi.height == 0)
    {
        greenScoreRoi = selectROI("tracker",frame);
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

extern "C" VideoAnalysis* find_video_touches(const char* video_path, uint8_t overlay_id)
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

extern "C" void js_memcpy(void* dest, void* source, size_t size) {
    memcpy(dest, source, size);
}

struct StreamBoutSegment
{
    int32_t start_frame;
    int32_t end_frame;
    const char* name_left;
    const char* name_right;
};

struct StreamAnalysis
{
    float64_t fps;
    int32_t frame_count;
    uint8_t bout_count;
    StreamBoutSegment bouts[64];
};

#include <tesseract/baseapi.h>

std::string formatString(const std::string& input) {
    size_t start = 0, end = input.size();

    // Trim from both ends
    while (start < end && !std::isalpha(input[start])) ++start;
    while (end > start && !std::isalpha(input[end - 1])) --end;

    if (start >= end) return ""; // Empty result if no valid characters

    // Allocate result string
    std::string result;
    result.reserve(end - start);

    // Normalize spaces and split into words
    std::vector<std::string> words;
    std::string currentWord;
    bool capitalize = true;

    for (size_t i = start; i < end; ++i) {
        if (std::isspace(input[i])) {
            if (!currentWord.empty()) {
                words.push_back(currentWord);
                currentWord.clear();
            }
            capitalize = true;
        } else {
            currentWord.push_back(capitalize ? std::toupper(input[i]) : std::tolower(input[i]));
            capitalize = false;
        }
    }
    if (!currentWord.empty()) {
        words.push_back(currentWord);
    }

    // If there's at least one word, rotate the first to the end
    if (!words.empty()) {
        std::rotate(words.begin(), words.begin() + 1, words.end());
    }

    // Reconstruct the result
    for (const auto& word : words) {
        if (!result.empty()) result.push_back(' ');
        result.append(word);
    }

    return result;
}

extern "C" StreamAnalysis* cut_stream(const char* ffmpeg_path, const char* tesseract_path, const char* svm_path, const char* video_path, uint8_t overlay_id, const char* output_folder)
{
    StreamAnalysis* analysis = new StreamAnalysis();

    cv::Ptr<cv::ml::SVM> svm = preload_digit_model(svm_path);

    tesseract::TessBaseAPI tess;
    if (tess.Init(tesseract_path, "eng", tesseract::OEM_LSTM_ONLY)) // Initialize Tesseract with English language
    {
        std::cerr << "Could not initialize Tesseract." << std::endl;
        return analysis;
    }
    tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);    // Set Page Segmentation Mode

    // How many seconds to skip at a time while waiting for a bout
    const int SKIP_SECONDS = 15;
    // Minimum length of bout (to prevent fluctuations in score)
    const int MIN_BOUT_SECONDS = 60;
    
    try
    {
        cv::VideoCapture cap(video_path, cv::CAP_FFMPEG);
        cap.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
        if (!cap.isOpened()) return 0;

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

        cv::Mat frame;

        int skip_rate = SKIP_SECONDS * fps;
        int min_bout_length = MIN_BOUT_SECONDS * fps;

        int frames_to_skip = 0;

        bool bout_running = false;
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
                std::cout << "Bout started at frame " << i << " (" << i / fps << "s, " << i * 100 / frame_count << "%)!" << std::endl;
                bout_running = true;
                analysis->bouts[analysis->bout_count].start_frame = i - skip_rate * 4;

                cv::Mat redName = frame(redNameRoi);
                cv::cvtColor(redName, redName, cv::COLOR_BGR2GRAY);
                tess.SetImage(redName.data, redName.cols, redName.rows, 1, redName.step); // Feed binary image to Tesseract
                std::string red_str = tess.GetUTF8Text(); // Extract text
                red_str = formatString(red_str);

                cv::Mat greenName = frame(greenNameRoi);
                cv::cvtColor(greenName, greenName, cv::COLOR_BGR2GRAY);
                tess.SetImage(greenName.data, greenName.cols, greenName.rows, 1, greenName.step); // Feed binary image to Tesseract
                std::string green_str = tess.GetUTF8Text(); // Extract text
                green_str = formatString(green_str);
                std::cout << "Red name: " << red_str << ", green name: " << green_str << std::endl;

                analysis->bouts[analysis->bout_count].name_left = strdup(red_str.c_str());
                analysis->bouts[analysis->bout_count].name_right = strdup(green_str.c_str());
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

        if (output_folder != NULL)
        {
            for (int i = 0; i < analysis->bout_count; i++)
            {
                cap.set(cv::CAP_PROP_POS_FRAMES, i);
                StreamBoutSegment bout = analysis->bouts[i];
                double start = bout.start_frame / fps;
                double end = bout.end_frame / fps;
                std::string bout_name = std::string("/") + bout.name_left + std::string(" vs ") + bout.name_right + ".mp4";
                std::cout << "Start: " << start << ", end: " << end << ", duration: " << end - start << std::endl;
                std::string command = std::string(ffmpeg_path) + " -loglevel quiet -ss " + std::to_string(start) + " -i \"" +
                                      std::string(video_path) + "\" -t " + std::to_string(end - start) +
                                      " -c copy -y \"" + std::string(output_folder) + bout_name + "\"";
                std::system(command.c_str());
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return analysis;
}
