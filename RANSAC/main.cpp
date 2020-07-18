#include<iostream>
#include <Dense>
#include <Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <pangolin/pangolin.h>
#include <unistd.h>
#include <vector>
#include <sophus/se3.h>
#include <string>
#include <boost/format.hpp>


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"


/*
使用RANSAC方法剔除无匹陪的特征点
 1. 提取特征点
 2. 利用所有的特征点计算两张图像的 变换矩阵H12   （使用  findHomography 函数，  需要设置阈值 deta）//   理解有误，  单应矩阵(Homography)H 的东西,它
    描述了两个平面之间的映射关系。若场景中的特征点都落在同一平面上  可以用Homography  矩阵
 3. 利用得到的H12 将第一张图像的所有特征点转换到第二张图像上
 4. 计算转换到第二张图像上之后的第一张图像的特征点与第二张图像上特征点的距离
 5. 距离小于 阈值deta(findHomography 函数 中的阈值) 的为内点 否则为 外点


 */
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
    int good_match=0;
    Mat obj=imread("./1.png",1);   //载入目标图像
    Mat scene=imread("./2.png",1); //载入场景图像
    if (obj.empty() || scene.empty() )
    {
        cout<<"Can't open the picture!\n";
        return 0;
    }
    vector<KeyPoint> obj_keypoints,scene_keypoints;
    Mat obj_descriptors,scene_descriptors;
    Ptr<FeatureDetector> detector =ORB::create();
//    ORB::detector detector;     //采用ORB算法提取特征点
    detector->detect(obj,obj_keypoints);
    detector->detect(scene,scene_keypoints);
    detector->compute(obj,obj_keypoints,obj_descriptors);
    detector->compute(scene,scene_keypoints,scene_descriptors);
    BFMatcher matcher(NORM_HAMMING,true); //汉明距离做为相似度度量
    vector<DMatch> matches;
    matcher.match(obj_descriptors, scene_descriptors, matches);
    Mat match_img;
    drawMatches(obj,obj_keypoints,scene,scene_keypoints,matches,match_img);
    imshow("滤除误匹配前",match_img);

    //保存匹配对序号
    vector<int> queryIdxs( matches.size() ), trainIdxs( matches.size() );
    for( size_t i = 0; i < matches.size(); i++ )
    {
        queryIdxs[i] = matches[i].queryIdx;
        trainIdxs[i] = matches[i].trainIdx;
    }

    Mat H12;   //变换矩阵

    vector<Point2f> points1; KeyPoint::convert(obj_keypoints, points1, queryIdxs);
    vector<Point2f> points2; KeyPoint::convert(scene_keypoints, points2, trainIdxs);
    int ransacReprojThreshold = 5;  //拒绝阈值


    H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );//  get the homography matrix from points2 to points1
    vector<char> matchesMask( matches.size(), 0 );
    Mat points1t;
    perspectiveTransform(Mat(points1), points1t, H12);   //transfotm the points to points1t using homography matrix H12
    for( size_t i1 = 0; i1 < points1.size(); i1++ )  //保存‘内点’
    {    // the points1t which is type Mat maybe is a column vector with points1.size() demension.
        if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= ransacReprojThreshold ) //给内点做标记
        {
            matchesMask[i1] = 1;
            good_match+=1;
        }
    }
    Mat match_img2;   //滤除‘外点’后
    drawMatches(obj,obj_keypoints,scene,scene_keypoints,matches,match_img2,Scalar::all(-1),Scalar::all(-1),matchesMask);// determine which keypoint will be draw based matchsMask

    //画出目标位置
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( obj.cols, 0 );
    obj_corners[2] = cvPoint( obj.cols, obj.rows ); obj_corners[3] = cvPoint( 0, obj.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H12);
    line( match_img2, scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),
          scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
    line( match_img2, scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),
          scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
    line( match_img2, scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),
          scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
    line( match_img2, scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),
          scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);

    imshow("滤除误匹配后",match_img2);
    cout<<"the number of good matchs"<<good_match<<endl;
    waitKey(0);

    return 0;
}


