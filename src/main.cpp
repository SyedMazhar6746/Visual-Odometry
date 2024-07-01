#include <iostream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <matplot/matplot.h>

namespace fs = std::filesystem;
namespace plt = matplot;


int main(){
// Setup paths
std::string data_path = "/root/dataset/data_odometry_gray/dataset/sequences/";
std::string sequence_num = "00/";
std::string left_images_dir = data_path + sequence_num + "image_0";
std::string right_images_dir =  data_path + sequence_num + "image_1";
int processImages = 1000;

// Vector to save all image paths
std::vector<std::string> left_image_paths;
std::vector<std::string> right_image_paths;

// Load all image paths
for (const auto & entry : fs::directory_iterator(left_images_dir)){
    left_image_paths.push_back(entry.path());
    }

for (const auto & entry : fs::directory_iterator(right_images_dir)){
    right_image_paths.push_back(entry.path());
    }

// Sort the vector of paths
std::sort(left_image_paths.begin(), left_image_paths.end());
std::sort(right_image_paths.begin(), right_image_paths.end());

// // Create a named window
// cv::namedWindow("SIFT Matches", cv::WINDOW_NORMAL);  // WINDOW_NORMAL allows resizing

// // Resize the window to the desired dimensions
// cv::resizeWindow("SIFT Matches", 1200, 600);  // Width: 1200 pixels, Height: 600 pixels

std::vector<cv::Mat> Transformations;

// 3D points in the global frame
std::vector<double> X;
std::vector<double> Y;
std::vector<double> Z;


cv::Mat T_init = cv::Mat::eye(4, 4, CV_64F);
Transformations.push_back(T_init);

cv::Mat P1 = (cv::Mat_<double>(3,4) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00,
                                        0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
                                        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);
    
cv::Mat P2 = (cv::Mat_<double>(3,4) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02,
                                        0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
                                        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);

cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
                                                    0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
                                                    0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);

// Feature detector
cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

// Match descriptors using FLANN matcher
cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
int step = 2;

// Start image processing
for (auto i = 0; i < left_image_paths.size() - step; i+=step){
    // Load images
    cv::Mat left_image_k = cv::imread(left_image_paths[i]);
    cv::Mat right_image_k = cv::imread(right_image_paths[i]);

    cv::Mat left_image_k1 = cv::imread(left_image_paths[i+step]);
    // cv::Mat right_image_k1 = cv::imread(right_image_paths[i+step]);

    // Check if the images were successfully loaded
    if (left_image_k.empty() || right_image_k.empty() || left_image_k1.empty()) { //|| right_image_k1.empty()
        std::cerr << "Error: Couldn't open the image file.\n";
        return 1;
    }

    std::vector<cv::KeyPoint> left_keypoints_k, right_keypoints_k, left_keypoints_k1, right_keypoints_k1;   
    cv::Mat left_descriptors_k, right_descriptors_k, left_descriptors_k1, right_descriptors_k1;

    // Compute features and descriptors at time k
    sift->detectAndCompute(left_image_k, cv::noArray(), left_keypoints_k, left_descriptors_k);
    sift->detectAndCompute(right_image_k, cv::noArray(), right_keypoints_k, right_descriptors_k);

    // Compute features and descriptors at time k + 1
    sift->detectAndCompute(left_image_k1, cv::noArray(), left_keypoints_k1, left_descriptors_k1);
    // sift->detectAndCompute(right_image_k1, cv::noArray(), right_keypoints_k1, right_descriptors_k1);


    // Match descriptors
    std::vector<std::vector<cv::DMatch>> knnMatches_k, knnMatches_k1;
    matcher->knnMatch(left_descriptors_k, right_descriptors_k, knnMatches_k, 2);
    matcher->knnMatch(left_descriptors_k, left_descriptors_k1, knnMatches_k1, 2);

    // Filter matches using ratio test
    std::vector<cv::DMatch> goodMatches_k, goodMatches_k1;
    float ratio_thresh = 0.85f;

    for (size_t i = 0; i < knnMatches_k.size(); i++) {
        if (knnMatches_k[i][0].distance < ratio_thresh * knnMatches_k[i][1].distance) {
            goodMatches_k.push_back(knnMatches_k[i][0]);
        }
    }

    for (size_t i = 0; i < knnMatches_k1.size(); i++) {
        if (knnMatches_k1[i][0].distance < ratio_thresh * knnMatches_k1[i][1].distance) {
            goodMatches_k1.push_back(knnMatches_k1[i][0]);
        }
    }


    // Satisfy the epipolar constraint i.e. select points which have similar y coordinate
    
    std::vector<cv::DMatch> refinedMatches;
    for(auto i = 0; i < goodMatches_k.size(); i++){
        auto left_image_y_coord = left_keypoints_k[goodMatches_k[i].queryIdx].pt.y;
        auto right_image_y_coord = right_keypoints_k[goodMatches_k[i].trainIdx].pt.y;
        auto diff = abs(left_image_y_coord - right_image_y_coord);
        if(diff < 0.2){
            // std::cout<< left_image_y_coord <<" "<< right_image_y_coord<< std::endl;
            refinedMatches.push_back(goodMatches_k[i]);
        }
    }
    goodMatches_k = refinedMatches;

    refinedMatches.clear();
    
    // Keep points with positive disparity
    
    for(auto i = 0; i < goodMatches_k.size(); i++){
        auto left_image_x_coord = left_keypoints_k[goodMatches_k[i].queryIdx].pt.x;
        auto right_image_x_coord = right_keypoints_k[goodMatches_k[i].trainIdx].pt.x;
        auto diff = left_image_x_coord - right_image_x_coord;
        if(diff > 0.0){
            // std::cout<< left_image_x_coord <<" "<< right_image_x_coord<< std::endl;
            refinedMatches.push_back(goodMatches_k[i]);
        }
    }
    goodMatches_k = refinedMatches;

    // for(auto i = 0; i < goodMatches_k.size(); i++){
    //     std::cout<< left_keypoints_k[goodMatches_k[i].queryIdx].pt.x << " " << left_keypoints_k[goodMatches_k[i].queryIdx].pt.y << std::endl;
    //     std::cout<< right_keypoints_k[goodMatches_k[i].trainIdx].pt.x << " " << right_keypoints_k[goodMatches_k[i].trainIdx].pt.y << std::endl;
    //     std::cout<< std::endl;
    // }

    std::vector<cv::DMatch> matchesAll_k, matchesAll_k1;
    // Get the matches present in left image at time k, right image at time k and left image at time k + 1
    for(auto i = 0; i < goodMatches_k.size(); i++){
        for(auto j = 0; j < goodMatches_k1.size(); j++){
            if(goodMatches_k[i].queryIdx == goodMatches_k1[j].queryIdx){
                // std::cout<<goodMatches_k[i].queryIdx << " " << goodMatches_k1[j].queryIdx << std::endl;
                // std::cout<<goodMatches_k[i].trainIdx << " " << goodMatches_k1[j].trainIdx << std::endl;
                // std::cout<<left_keypoints_k[goodMatches_k[i].queryIdx].pt.x<<" "<<left_keypoints_k[goodMatches_k[i].queryIdx].pt.y<<std::endl;
                // std::cout<<right_keypoints_k[goodMatches_k[i].trainIdx].pt.x<<" "<<right_keypoints_k[goodMatches_k[i].trainIdx].pt.y<<std::endl;
                // std::cout<<left_keypoints_k[goodMatches_k1[j].queryIdx].pt.x<<" "<<left_keypoints_k[goodMatches_k1[j].queryIdx].pt.y<<std::endl;
                // std::cout<<left_keypoints_k1[goodMatches_k1[j].trainIdx].pt.x<<" "<<left_keypoints_k1[goodMatches_k1[j].trainIdx].pt.y<<std::endl;
                // std::cout<<std::endl;
                matchesAll_k.push_back(goodMatches_k[i]);
                matchesAll_k1.push_back(goodMatches_k1[j]);
                break;
            }
        }
    }

    std::vector<cv::Point2f> points1, points2;
    for(auto i = 0; i < matchesAll_k.size(); i++){
        points1.push_back(left_keypoints_k[matchesAll_k[i].queryIdx].pt);
        points2.push_back(right_keypoints_k[matchesAll_k[i].trainIdx].pt);
    }      

    // Triangulate points
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, points1, points2, points4D);

    // Convert homogeneous coordinates to Cartesian coordinates
    std::vector<cv::Point3f> points3D;
    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat col = points4D.col(i);
        col /= col.at<float>(3); // Normalize by the fourth coordinate (homogeneous coordinate)
        cv::Point3f pt(col.at<float>(0), col.at<float>(1), col.at<float>(2));
        points3D.push_back(pt);
    }

    /*Stuff for plotting all 3d points in global frame -> not working for the moment*/
    // // Change type of points4D from CV_32F to CV_64F
    // cv::Mat points4DConverted;
    // points4D.convertTo(points4DConverted, CV_64F);

    // // Transform the 3D points to global coordinate system
    // cv::Mat transformed4D;
    // transformed4D = Transformations.back() * points4DConverted;

    // // Convert the points from homogeneous to cartesian coordinates
    // // cv::Mat points3DCart;
    // for (int i = 0; i < transformed4D.cols; i++) {
    //     cv::Mat col = transformed4D.col(i);
    //     col /= col.at<float>(3); // Normalize by the fourth coordinate (homogeneous coordinate)
    //     X.push_back(col.at<float>(0));
    //     Y.push_back(col.at<float>(1));
    //     Z.push_back(col.at<float>(2));
    //     // cv::Point3f pt(col.at<float>(0), col.at<float>(1), col.at<float>(2));
    //     // points3DCart.push_back(pt);
    // }

    // std::cout<<X.size()<<std::endl;
    // matplot::plot3(X, Y, Z, "o");
    // matplot::show();
    // break;
    /*===========================================================================*/

    // Define maximum allowable distance
    float max_distance = 40.0; // Adjust this value according to your requirements

    // Filter points that are too far
    std::vector<cv::Point3d> filteredPoints3D;
    std::vector<cv::DMatch> filteredPoints_k1;
    for (auto i = 0; i<points3D.size(); i++) {
        if (norm(points3D[i]) <= max_distance) {
            filteredPoints3D.push_back(points3D[i]);
            filteredPoints_k1.push_back(matchesAll_k1[i]);
        }
    }

    // for(const auto& pt: filteredPoints3D){
    //     std::cout<<pt<<std::endl;
    // }

    // Get the image coorinates of the 3D points in the frame at k + 1
    std::vector<cv::Point2d> matchesFrame_k1;
    for(auto i = 0; i<filteredPoints_k1.size(); i++){
        matchesFrame_k1.push_back(left_keypoints_k1[filteredPoints_k1[i].trainIdx].pt);
    }

    cv::Mat rvec, tvec;
    bool useExtrinsicGuess = false;
    int iterationsCount = 100;
    float reprojectionError = 8.0;
    double confidence = 0.99;
    cv::Mat inliers;
    int flags = cv::SOLVEPNP_SQPNP; 
    if(filteredPoints3D.size()>=4){
        cv::solvePnPRansac(filteredPoints3D, matchesFrame_k1, cameraMatrix, cv::Mat(), rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers, flags);

        // Convert rotation vector to rotation matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // Create transformation matrix
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(T(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(T(cv::Rect(3, 0, 1, 3)));

        // Output the transformation matrix
        // std::cout << "Transformation matrix:\n" << T << std::endl;
        cv::Mat inverseT;
        cv::invert(T, inverseT);
        cv::Mat newPose = Transformations.back() * inverseT;
        Transformations.push_back(newPose);
    }

    // // Draw matches
    // cv::Mat img_matches;
    // cv::drawMatches(left_image_k, left_keypoints_k, right_image_k, right_keypoints_k, matchesAll_k, img_matches);

    // // Display matches
    // cv::imshow("SIFT Matches", img_matches);
    // cv::waitKey(5);

    std::cout<<"Images processed: "<<i<<std::endl;
    if(i == processImages){
        break;
    }

}

std::vector<double> x;
std::vector<double> z;
for(const auto& T: Transformations){
    auto xCoord = T.at<double>(0, 3);
    auto zCoord = T.at<double>(2, 3);
    x.push_back(xCoord);
    z.push_back(zCoord);
}

// Plot the points using matplotlib-cpp
plt::plot(x, z, "-");
plt::xlabel("X");
plt::ylabel("Z");
plt::title("Trajectory XZ");
plt::show();

return 0;
}
