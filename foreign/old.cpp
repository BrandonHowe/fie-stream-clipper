#include <cstdint>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

extern "C" int32_t add(int32_t a, int32_t b) {
    return a + b;
}

extern "C" int32_t find_video_touches(const char* video_path)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) return 0;

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::cout << "FPS: " << fps << "\n"
            << "Width: " << width << "\n"
            << "Height: " << height << "\n"
            << "Total Frames: " << frame_count << std::endl;

    cv::Rect roi;
    cv::Mat frame;
    cv::Ptr<cv::Tracker> tracker = cv::TrackerMIL::create();

    cap >> frame;
    // roi= selectROI("tracker",frame);

    // if (roi.width == 0 || roi.height == 0) return 0;

    // tracker->init(frame, roi);

    printf("Start the tracking process, press ESC to quit.\n");

    cap.set(cv::CAP_PROP_POS_FRAMES, 300);
    // cap >> frame;

    // cv::Mat redOnly;

    // cv::inRange(frame, cv::Scalar(0, 0, 150), cv::Scalar(120, 120, 255), redOnly);

    // std::vector<std::vector<cv::Point>> contours;
    // cv::findContours(redOnly, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // for (const auto& contour : contours) {
    //     cv::Rect boundRect = cv::boundingRect(contour);
    //     double aspectRatio = static_cast<double>(boundRect.width) / boundRect.height;

    //     if (cv::contourArea(contour) > 100 && aspectRatio > 0.8 && aspectRatio < 1.2) {  // Adjust area and aspect ratio as needed
    //         cv::rectangle(redOnly, boundRect, cv::Scalar(0, 255, 0), 2);  // Draw green rectangles around detected red rectangles
    //     }
    // }

    imshow("tracker", frame);

    // cv::waitKey(10000);

    std::vector<int> red_light_timestamps;
    bool can_score_left = true;
    int touch_left_count = 0;

    // return 1000;
    for (int i = 0; i < frame_count; i++) {
        // get frame from the video
        cap >> frame;
        cv::Mat hsvFrame;
        cv::cvtColor(frame, hsvFrame, cv::COLOR_RGB2HSV);
        // if (i % 100 == 0) std::cout << "Frame " << i << std::endl;
        while (red_light_timestamps.size() > 0 && red_light_timestamps[0] < i - 100)
        {
            red_light_timestamps.erase(red_light_timestamps.begin());
        }
        if (red_light_timestamps.size() < 20) can_score_left = true;
        if (red_light_timestamps.size() > 20 && can_score_left) {
            can_score_left = false;
            std::cout << "Touch left " << ++touch_left_count << " at " << i << std::endl;
        }
    
        // stop the program if no more images
        if(frame.rows==0 || frame.cols==0) break;
        
        cv::Mat redMask1, redMask2, redMask, redMaskAnd;
        cv::Mat greenMask, redEdges, greenEdges;
        cv::inRange(hsvFrame, cv::Scalar(120, 0, 190), cv::Scalar(135, 128, 255), redMask1);
        cv::inRange(hsvFrame, cv::Scalar(0, 0, 250), cv::Scalar(180, 5, 255), redMask2);
        cv::bitwise_or(redMask1, redMask2, redMask);
        // cv::bitwise_not(redMask2, redMask2);
        cv::bitwise_and(frame, frame, redMaskAnd, redMask);
        // cv::inRange(hsvFrame, cv::Scalar(0, 0, 180), cv::Scalar(130, 130, 255), redMask1);
        // cv::inRange(hsvFrame, cv::Scalar(240, 240, 240), cv::Scalar(255, 255, 255), redMask2);
        cv::inRange(frame, cv::Scalar(50, 180, 150), cv::Scalar(130, 255, 255), greenMask);

        cv::Canny(redMask, redEdges, 50, 150);
        // cv::Canny(greenMask, greenEdges, 50, 150);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(redEdges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours)
        {
            cv::Rect rect = cv::boundingRect(contour);
            if (rect.width > rect.height && rect.width < rect.height * 3 && rect.area() > 70 && rect.area() < 1000)
            {
                cv::Mat roi = redMask(rect);

                std::vector<std::vector<cv::Point>> roiContours;
                cv::findContours(roi, roiContours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
                if (roiContours.size() > 8) {
                    cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
                    red_light_timestamps.push_back(i);
                    // std::cout << rect.area() << std::endl;
                    // if (can_score_left) {
                        std::cout << "Red light on at " << rect.width << "|" << rect.height << " (total: " << red_light_timestamps.size() << ")" << std::endl;
                        // can_score_left = false;
                    // }
                }
            }
        }

        // cv::findContours(greenEdges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // for (const auto& contour : contours)
        // {
        //     cv::Rect rect = cv::boundingRect(contour);
        //     if (rect.width > rect.height && rect.area() > 100 && rect.area() < 1000 && rect.height * 3 > rect.width)
        //     {
        //         cv::Mat roi = greenMask(rect);

        //         std::vector<std::vector<cv::Point>> roiContours;
        //         cv::findContours(roi, roiContours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        //         if (roiContours.size() > 1) {
        //             cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);
        //             // std::cout << "Green light on at " << i << std::endl;
        //         }
        //     }
        // }
    
        // tracker->update(frame,roi);
    
        // rectangle( frame, roi, cv::Scalar( 255, 0, 0 ), 2, 1 );
    
        imshow("tracker", redMaskAnd);
    
        //quit on ESC button
        if(cv::waitKey(1)==27)break;
    }

    return 1000;
}