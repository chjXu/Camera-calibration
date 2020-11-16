#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <OpenNI.h>
#include <opencv2/aruco.hpp>
#include <boost/thread.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <pthread.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <depth_registration.h>
#include <fstream>
#include <math.h>

#include <openpose_ros_msgs/OpenPoseHumanList.h>
#include <openpose_ros_msgs/PointWithProb.h>

#include <inertial_poser/ROI_Package.h>
#include <inertial_poser/UpperBodyKeyPoints.h>
#include <inertial_poser/KeyPoint.h>

#include <geometry_msgs/PoseArray.h>
using namespace cv;
using namespace std;

typedef struct keypoints_with_prob
{
  double x;
  double y;
  double p;
}KeyPoints;

void visualize_depth(Mat& depth_image, Mat& depth_viz)
{
   if(!depth_image.empty())
   {
       depth_viz = Mat(depth_image.rows, depth_image.cols, CV_8UC3);
       for (int r = 0; r < depth_viz.rows; ++r)
        for (int c = 0; c < depth_viz.cols; ++c)
        {
            uint16_t depth = depth_image.at<uint16_t>(r, c);
            uint16_t level;
            uint8_t alpha;

            //sort depth information into different depth levels
            if (depth == 0)
                level = 0;
            else
                level = depth / 1000 + 1;
                alpha = (depth % 1000) / 4;

            switch(level)
            {
                case(1):

                    depth_viz.at<Vec3b>(r, c) = Vec3b(0, 0, alpha);
                    break;
                case(2):
                    depth_viz.at<Vec3b>(r, c) = Vec3b(0, alpha, 255);
                    break;
                case(3):
                    depth_viz.at<Vec3b>(r, c) = Vec3b(0, 255, 255-alpha);
                    break;
                case(4):
                    depth_viz.at<Vec3b>(r, c) = Vec3b(alpha, 255, 0);
                    break;
                case(5):
                    depth_viz.at<Vec3b>(r, c) = Vec3b(255, 255-alpha, 0);
                    break;
                default:
                    depth_viz.at<Vec3b>(r, c) = Vec3b(0, 0, 0);
                    break;
           }

        }
   }
}


class ImageProcessor
{
  public:
    ImageProcessor(bool rect, bool isCalibrationMode):it(nh), sizeColor(1920, 1080), max_queue_size(5) //left_key_points_center(480,260) ,right_key_points_center(480,260), left_ROI(240), right_ROI(240)
    {


      string prime_color_topic = "/kinect2_1/qhd/image_color";
      string sub_color_topic = "/kinect2_2/qhd/image_color";
      if(rect){
        prime_color_topic = prime_color_topic + "_rect";
        sub_color_topic = sub_color_topic + "_rect";
      }
      prime_color_sub = it.subscribe(prime_color_topic.c_str(), 1,&ImageProcessor::prime_imageCallback,this);
      sub_color_sub = it.subscribe(sub_color_topic.c_str(), 1,&ImageProcessor::sub_imageCallback,this);
      //depth_sub = it.subscribe("/kinect2_1/qhd/image_depth_rect",1,&ImageProcessor::depthimageCallback,this);

      //spliced_image_pub = it.advertise("spliced", 1);

      //keypoints_pub = nh.advertise<inertial_poser::ROI_Package>("/inertial_poser/roi_package", 5);
      human_keypoints_sub = nh.subscribe("/human_1/roi_package", 1, &ImageProcessor::human_keypoints_callback, this);

      //human_joint_sub = nh.subscribe("inertial_poser/pose2d",1,&ImageProcessor::human_joint_callback,this);


      //human_keypoints_sub = nh.subscribe("/openpose_ros/human_list", 1, &ImageProcessor::human_keypoints_callback, this);

        joint_names.push_back("shoulder_link");
        joint_names.push_back("upper_arm_link");
	      joint_names.push_back("forearm_left");
        joint_names.push_back("forearm_link");
        joint_names.push_back("wrist_1_link");
        joint_names.push_back("wrist_2_link");
        joint_names.push_back("wrist_3_link");

        human_joint_names.push_back("hip");
        human_joint_names.push_back("spine");
        human_joint_names.push_back("spine1");
        human_joint_names.push_back("spine2");
        human_joint_names.push_back("spine3");
        human_joint_names.push_back("lShoulder");
        human_joint_names.push_back("lArm");
        human_joint_names.push_back("lForeArm");
        human_joint_names.push_back("lWrist");
        human_joint_names.push_back("lHand");
        human_joint_names.push_back("rShoulder");
        human_joint_names.push_back("rArm");
        human_joint_names.push_back("rForeArm");
        human_joint_names.push_back("rWrist");
        human_joint_names.push_back("rHand");

        marker_0_sum_count = 0;

      isCamBaseTransformAvailable = false;
      isCamHumanTransformAvailable = false;
      firstDepth = true;
      calibrationMode = isCalibrationMode;
      if(!calibrationMode)
      {
        string pose_solution_path = "/home/agent/luk_ws/robot_pose/solution_20180814";
        loadRobotPoseFile(pose_solution_path);
      }

      //pthread_mutex_init (&mutex, NULL);
    }
    bool getImage(Mat&);

    void prime_imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void sub_imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthimageCallback(const sensor_msgs::ImageConstPtr& msgdepth);
    void loadCalibrationFiles(string& calib_path, cv::Mat& cameraMatrix, cv::Mat& distortion, double scale);
    void initCalibration();
    void sendMarkerTF(vector<Vec3d>& marker_trans, vector<Vec3d>& marker_rot, vector<int>& ids);
    void sendCameraTF(vector<Vec3d>& marker_trans, vector<Vec3d>& marker_rot, vector<int>& ids, bool prime);
    void sendMarkerTf(vector<Point3f>& marker_position, vector<int>& ids);
    void getWorldCoordinate(Point2f& image_cord, Point3f& cord, cv::Mat&);
    void calculateRobotPose(vector<Point>& joint_image_cords, vector<Point3f>& joint_3d_cords, bool prime);
    void getImageCoordinate(Point3f& world_cord, Point& image_cord, cv::Mat&);
    void drawRobotJoints(Mat& image, vector<Point>& joint_image_cords);
    void linkToRobotTf();

    void human_joint_callback(const geometry_msgs::PoseArray& poses);
    void calculateHumanPose(vector<Point>& joint_image_cords, vector<Point3f>& joint_3d_cords, bool);
    void draw_human_pose(Mat& image, vector<Point>& human_joints);
    //void getFivemarkerWorldCoordinate(vector<vector<Point2f>>& corners, vector<int>& ids, vector<Point2f>& marker_center, vector<Point3f>& world_cord);
    bool removeRobotImage(Mat& image,vector<Point>& robot_image_pos, vector<Point3f>& robot_image_3d_pos, Mat& );
    bool drawPointPiexs(Mat& image,vector<Point>& robot_image_pos, vector<Point3f>& robot_image_3d_pos, Mat&, int n);

    static void* publishRobotThread(void *arg);
    void startRobotThread();
    void stopRobotThread();

    void publishSplicedImage();
    static void* publishSplicedImageThread(void *arg);
    void startSplicedThread();
    void stopSplicedThread();

    void displayImg();

    void human_keypoints_callback(inertial_poser::ROI_Package roi_pack);
    void drawKeyPoints(Mat& image, vector<KeyPoints>& points);

    void drawGoalPoint(Mat& image, bool prime);
  private:
    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    image_transport::Subscriber prime_color_sub;
    image_transport::Subscriber depth_sub;
    image_transport::Subscriber sub_color_sub;
    image_transport::Subscriber subTF;
    image_transport::Publisher spliced_image_pub;

    ros::Subscriber human_joint_sub;
    ros::Subscriber human_keypoints_sub;
    ros::Publisher keypoints_pub;
    cv::Mat distortion_prime;
    cv::Mat cameraMatrix_prime;
    cv::Mat distortion_sub;
    cv::Mat cameraMatrix_sub;
    Point2f averagecorners;

    //vector<Point> robot_image_pos;
    //vector<Point3f>robot_image_3d_pos;

    Vec3d marker_0_tvecs_sum;
    Vec3d marker_0_rvecs_sum;
    int marker_0_sum_count;

    //cv::Mat color_mat;
    vector<cv::Mat> prime_color_mat_queue;
    vector<cv::Mat> depth_mat_queue;
    vector<cv::Mat> sub_color_mat_queue;
    int max_queue_size;
    //cv::Mat displyMat_prime;
    //cv::Mat displyMat_sub;


    tf::TransformListener robot_pose_listener;
    tf::StampedTransform base_cam_transform;
    bool isCamBaseTransformAvailable;
    bool isCamHumanTransformAvailable;
    vector<std::string> joint_names;
    vector<std::string> human_joint_names;
    Size sizeColor;
    bool calibrationMode;
    bool firstDepth;

    vector<Point> human_joint_pos;

    tf::Transform robot_pose_tansform;
    void loadRobotPoseFile(string);

    vector<KeyPoints> keypoints_prime;
    vector<KeyPoints> keypoints_sub;
    bool keypoints_available;

};

void ImageProcessor::human_keypoints_callback(inertial_poser::ROI_Package roi_pack)
{
        KeyPoints keypoint_element;
        keypoints_prime.clear();
        keypoints_sub.clear();
        for(int i = 0; i < 7; i++)
        {
          keypoint_element.x = roi_pack.packages[0].points[i].x;
          keypoint_element.y = roi_pack.packages[0].points[i].y;
          keypoint_element.p = roi_pack.packages[0].points[i].prob;
          keypoint_element.p *= keypoint_element.p * keypoint_element.p;
          keypoints_prime.push_back(keypoint_element);

          keypoint_element.x = roi_pack.packages[1].points[i].x;
          keypoint_element.y = roi_pack.packages[1].points[i].y;
          keypoint_element.p = roi_pack.packages[1].points[i].prob;
          keypoint_element.p *= keypoint_element.p * keypoint_element.p;
          keypoints_sub.push_back(keypoint_element);
        }

        keypoints_available = true;
        ROS_INFO("keypoints available");
}

void ImageProcessor::initCalibration(){
    string calib_path_prime = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/003415165047";
    string calib_path_sub = "/home/agent/catkin_ws/src/iai_kinect2/kinect2_bridge/data/092465240847";
    loadCalibrationFiles(calib_path_prime, cameraMatrix_prime, distortion_prime, 0.5);
    loadCalibrationFiles(calib_path_sub, cameraMatrix_sub, distortion_sub, 0.5);
}

void ImageProcessor::drawGoalPoint(Mat& image, bool prime)
{
  string reference_frame = "camera_base";
  if(prime)
  {
    reference_frame = reference_frame + "_1";
  }
  else
  {
    reference_frame = reference_frame + "_2";
  }
  tf::StampedTransform desired_transform;
  tf::StampedTransform rect_transform;

  try
  {
      robot_pose_listener.lookupTransform(reference_frame.c_str(), "desired_goal", ros::Time(0), desired_transform);
      robot_pose_listener.lookupTransform(reference_frame.c_str(), "rect_goal", ros::Time(0), rect_transform);
      Point3f desired_point3d(desired_transform.getOrigin().x(),
                              desired_transform.getOrigin().y(),
                              desired_transform.getOrigin().z());

      Point3f rect_point3d(rect_transform.getOrigin().x(),
                           rect_transform.getOrigin().y(),
                           rect_transform.getOrigin().z());
      Point desired_point2d, rect_point2d;
      if(prime){
        getImageCoordinate(desired_point3d, desired_point2d, cameraMatrix_prime);
        getImageCoordinate(rect_point3d, rect_point2d, cameraMatrix_prime);
      }
      else{
        getImageCoordinate(desired_point3d, desired_point2d, cameraMatrix_sub);
        getImageCoordinate(rect_point3d, rect_point2d, cameraMatrix_sub);
      }
      circle(image, cv::Point((int)desired_point2d.x, (int)desired_point2d.y), 3, Scalar(255, 0, 255), -1, 8);
      circle(image, cv::Point((int)rect_point2d.x, (int)rect_point2d.y), 3, Scalar(0, 255, 255), -1, 8);

      //putText(image, prob_str.c_str(), cv::Point((int)points[i].x - 30, (int)points[i].y - 10), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0));
  }
  catch(tf::TransformException ex)
  {
      //ROS_ERROR("%s", ex.what());
      return;
  }

}


void ImageProcessor::drawKeyPoints(Mat& image, vector<KeyPoints>& points)
{
  //circle(image, human_joints[0], 3, Scalar(0, 0, 255), -1, 8);
  //draw joints on image
  for(int i = 0; i < (points.size()); i++)
  {
      if(points[i].p > 0){
        stringstream ss;
        string prob_str;
        circle(image, cv::Point((int)points[i].x, (int)points[i].y), 10, Scalar(255, 0, 0), 3, 8);
        //cout << points[i].p;
        ss << points[i].p;
        ss >> prob_str;

        putText(image, prob_str.c_str(), cv::Point((int)points[i].x - 60, (int)points[i].y - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0));
      }
  }
}
void ImageProcessor::human_joint_callback(const geometry_msgs::PoseArray& poses)
{
  cv::Point joint_pos;
  human_joint_pos.clear();
  for(int i = 0;i < poses.poses.size(); ++i)
  {
    joint_pos.x = poses.poses[i].position.x;
    joint_pos.y = poses.poses[i].position.y;
    human_joint_pos.push_back(joint_pos);
  }
}
void ImageProcessor::calculateHumanPose(vector<Point>& joint_image_cords, vector<Point3f>& joint_3d_cords, bool prime)
{
  //tf::TransformListener robot_pose_listener;
    string human_reference_frame;

    human_reference_frame = "camera_base";
    if(prime)
    {
      human_reference_frame = human_reference_frame + "_1";
    }
    else
    {
      human_reference_frame = human_reference_frame + "_2";
    }



    tf::StampedTransform joint_transforms;
    tf::StampedTransform cam_hip_transform;
    try
    {
        robot_pose_listener.lookupTransform(human_reference_frame.c_str(), "/human_1/hip", ros::Time(0), cam_hip_transform);
    }

    catch(tf::TransformException ex)
    {
        //ROS_ERROR("%s", ex.what());
        isCamHumanTransformAvailable = false;
        return;
    }

    isCamHumanTransformAvailable = true;
    Point3f hip_location(cam_hip_transform.getOrigin().x(), cam_hip_transform.getOrigin().y(), cam_hip_transform.getOrigin().z());
    Point hip_image_cord;
    if(prime){
      getImageCoordinate(hip_location, hip_image_cord, cameraMatrix_prime);
    }
    else{
      getImageCoordinate(hip_location, hip_image_cord, cameraMatrix_sub);
    }
    //circle(color_mat, hip_image_cord, 2, Scalar(0, 0, 255), -1, 8);
    /***
    ostringstream cord_text;
    cord_text.str("");
    cord_text << "base_position:" << " at" << '(' << base_location.x << ',' << base_location.y << ',' << base_location.z << ')';
    putText(color_mat, cord_text.str(), Point(20,400), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,, 0));
    ***/
    //vector<Point3f> joint_3d_cords;
    joint_3d_cords.push_back(hip_location);
    //vector<Point> joint_image_cords;
    joint_image_cords.push_back(hip_image_cord);
    for(int i = 1; i < human_joint_names.size(); i++)
    {
        try
        {
            robot_pose_listener.lookupTransform(human_reference_frame.c_str(), "/human_1/" + human_joint_names[i], ros::Time(0), joint_transforms);
        }
        catch(tf::TransformException ex)
        {
            //ROS_ERROR("%s", ex.what());
            continue;
        }
        Point3f location(joint_transforms.getOrigin().x(), joint_transforms.getOrigin().y(), joint_transforms.getOrigin().z());
        joint_3d_cords.push_back(location);
        Point joint_image_cord;
        if(prime){
          getImageCoordinate(location, joint_image_cord, cameraMatrix_prime);
        }
        else{
          getImageCoordinate(location, joint_image_cord, cameraMatrix_sub);
        }
        joint_image_cords.push_back(joint_image_cord);

    }
    //ROS_INFO("Human Pose calculated");
}
void ImageProcessor::draw_human_pose(Mat& image, vector<Point>& human_joints)
{
  circle(image, human_joints[0], 3, Scalar(0, 0, 255), -1, 8);
  //draw joints on image
  for(int i = 1; i < (human_joints.size()); i++)
  {
      circle(image, human_joints[i], 3, Scalar(0, 0, 255), -1, 8);
      if (i != 10)
        line(image,human_joints[i-1], human_joints[i], Scalar(0, 0, 255), 2);
      else
        line(image,human_joints[5], human_joints[i], Scalar(0, 0, 255), 2);
  }
    //ROS_INFO("Human Pose drawed");
}


void ImageProcessor::drawRobotJoints(Mat& image, vector<Point>& joint_image_cords)
{

    //cout << "x: " << joint_image_cords[0].x << "\n";
    //cout << "y: " << joint_image_cords[0].y << "\n";
    circle(image, joint_image_cords[0], 3, Scalar(0, 255, 0), -1, 8);
    //draw joints on image
    for(int i = 1; i < (joint_image_cords.size()); i++)
    {
        circle(image, joint_image_cords[i], 3, Scalar(0, 255, 0), -1, 8);
        line(image,joint_image_cords[i-1],joint_image_cords[i], Scalar(0, 255, 255), 2);
    }
}
void ImageProcessor::getImageCoordinate(Point3f& world_cord, Point& image_cord, Mat& cameraMatrix)
{
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    image_cord.x = (int)(world_cord.x * fx / world_cord.z + cx);
    image_cord.y = (int)(world_cord.y * fy / world_cord.z + cy);
}

void ImageProcessor::calculateRobotPose(vector<Point>& joint_image_cords, vector<Point3f>& joint_3d_cords, bool prime = true)
{
  //tf::TransformListener robot_pose_listener;
    string robot_reference_frame;
    if (calibrationMode)
    {
        robot_reference_frame = "camera_base_rect";
    }
    else
    {
        robot_reference_frame = "camera_base";
    }

    if(prime)
    {
      robot_reference_frame = robot_reference_frame + "_1";
    }
    else
    {
      robot_reference_frame = robot_reference_frame + "_2";
    }


    tf::StampedTransform joint_transforms;
    tf::StampedTransform cam_base_transform;
    try
    {
        robot_pose_listener.lookupTransform(robot_reference_frame.c_str(), "base_link", ros::Time(0), cam_base_transform);
    }

    catch(tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
        isCamBaseTransformAvailable = false;
        return;
    }

    isCamBaseTransformAvailable = true;
    Point3f base_location(cam_base_transform.getOrigin().x(), cam_base_transform.getOrigin().y(), cam_base_transform.getOrigin().z());
    Point base_image_cord;
    if(prime){
      getImageCoordinate(base_location, base_image_cord, cameraMatrix_prime);
    }
    else{
      getImageCoordinate(base_location, base_image_cord, cameraMatrix_sub);
    }
    //circle(color_mat, base_image_cord, 2, Scalar(0, 255, 255), -1, 8);

    ostringstream cord_text;
    cord_text.str("");
    cord_text << "base_position:" << " at" << '(' << base_location.x << ',' << base_location.y << ',' << base_location.z << ')';
    //putText(color_mat, cord_text.str(), Point(20,400), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255, 0));

    //vector<Point3f> joint_3d_cords;
    joint_3d_cords.push_back(base_location);
    //vector<Point> joint_image_cords;
    joint_image_cords.push_back(base_image_cord);
    for(int i = 0; i < joint_names.size(); i++)
    {
        try
        {
            robot_pose_listener.lookupTransform(robot_reference_frame.c_str(), joint_names[i], ros::Time(0), joint_transforms);
        }
        catch(tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
            continue;
        }
        Point3f location(joint_transforms.getOrigin().x(), joint_transforms.getOrigin().y(), joint_transforms.getOrigin().z());
        joint_3d_cords.push_back(location);
        Point joint_image_cord;
        if(prime){
          getImageCoordinate(location, joint_image_cord, cameraMatrix_prime);
        }
        else{
          getImageCoordinate(location, joint_image_cord, cameraMatrix_sub);
        }
        joint_image_cords.push_back(joint_image_cord);
        //ROS_INFO("Robot Pose Get!");

    }

}
void ImageProcessor::loadCalibrationFiles(string& calib_path, cv::Mat& cameraMatrix, cv::Mat& distortion, double scale)
{

    cv::FileStorage fs;

    cv::Mat cameraMatrix_origin;


  if(fs.open(calib_path + "/calib_color.yaml", cv::FileStorage::READ))
  {
    fs["cameraMatrix"] >> cameraMatrix_origin;
    cameraMatrix = cameraMatrix_origin.clone();
    cameraMatrix.at<double>(0, 0) *= scale;
    cameraMatrix.at<double>(1, 1) *= scale;
    cameraMatrix.at<double>(0, 2) *= scale;
    cameraMatrix.at<double>(1, 2) *= scale;

    distortion= cv::Mat::zeros(1, 5, CV_64F);

    //fs["distortionCoefficients"] >> distortion_color;
    cout << "color matrix load success"<< endl;
    fs.release();


  }
  else
  {
    cout << "No calibration file: calib_color.yalm, using default calibration setting" << endl;
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distortion = cv::Mat::zeros(1, 5, CV_64F);


  }


}
void ImageProcessor::getWorldCoordinate(Point2f& image_cord, Point3f& cord,Mat& cameraMatrix)
{

    if(!prime_color_mat_queue.empty() && !depth_mat_queue.empty() && image_cord.x < sizeColor.width && image_cord.y < sizeColor.height)
    {
        double fx = cameraMatrix.at<double>(0, 0);
        double fy = cameraMatrix.at<double>(1, 1);
        double cx = cameraMatrix.at<double>(0, 2);
        double cy = cameraMatrix.at<double>(1, 2);

        uint16_t d = depth_mat_queue[0].at<uint16_t>(image_cord);

        cord.z = float(d) * 0.001f;
        //printf("%.4f\n", cord.z);
        cord.x = ((image_cord.x - cx) * cord.z) / fx;
        cord.y = ((image_cord.y - cy) * cord.z) / fy;
    }
}

void ImageProcessor::depthimageCallback(const sensor_msgs::ImageConstPtr& msgdepth)
{

	try
    {
      vector<int> markerfiveids(1,5);

    	Mat depth_image = cv_bridge::toCvShare(msgdepth, "16UC1")->image;
      depth_mat_queue.push_back(depth_image);

      Mat depth_viz;
      visualize_depth(depth_mat_queue[0], depth_viz);
      //cv::imshow("depth", depth_viz);
      waitKey();
    	//cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    	//detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
       // cv::Mat displyImg = cv_bridge::toCvShare(msgdepth, "mono16")->image.clone();
        //Point2f marker_center;
        vector<Point3f> world_cord;
        //marker_center.push_back(averagecorners);
        Point3f cord3(0.f, 0.f, 0.f);
        getWorldCoordinate(averagecorners, cord3, cameraMatrix_prime);
        world_cord.push_back(cord3);
        sendMarkerTf(world_cord, markerfiveids);

    }
    catch (cv_bridge::Exception& e)
    {
    	ROS_ERROR("Could not convert from '%s' to 'mono16'.", msgdepth->encoding.c_str());
    }
}

void ImageProcessor::prime_imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
      //cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
      Mat color_mat = cv_bridge::toCvShare(msg, "bgr8")->image;
      //pthread_mutex_lock(&mutex);
      prime_color_mat_queue.push_back(color_mat.clone());
      //cout << "prime queue size: " << prime_color_mat_queue.size() << "\n";
      if(prime_color_mat_queue.size() > max_queue_size){

        prime_color_mat_queue.erase(prime_color_mat_queue.begin());
      }
      //pthread_mutex_unlock(&mutex);
      Mat displyImg = prime_color_mat_queue[0].clone();


      vector<Point> joint_image_cords;
      vector<Point3f> joint_3d_cords;
      //namedWindow("Color Frame");
      vector<Point> human_image_cords;
      vector<Point3f> human_3d_cords;

      if(!prime_color_mat_queue.empty())
      {
        calculateRobotPose(joint_image_cords, joint_3d_cords, true);
        if(!joint_image_cords.empty())
        {
          //Mat depth_debug;
          //depth_debug = depth_mat.clone();
          //if(!removeRobotImage(displyImg, joint_image_cords, joint_3d_cords, depth_debug))
            //imshow("origin depth", depth_mat);
          drawRobotJoints(displyImg,joint_image_cords);
        }
        calculateHumanPose(human_image_cords, human_3d_cords, true);
        if(!human_image_cords.empty())
          draw_human_pose(displyImg, human_image_cords);
        //imshow("Color Frame", color_mat);
        if(!keypoints_prime.empty())
          drawKeyPoints(displyImg, keypoints_prime);

        drawGoalPoint(displyImg, true);
      }


/**
      cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

      Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
      //detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
      detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;
      //cv::Mat displyImg = color_mat.clone();
      //flip(displyImg,displyImg,1);
      std::vector< int > ids;
      std::vector< std::vector< cv::Point2f > > corners, rejected;

      //std::vector<cv::Point3f> world_cord;
      //  vector< Vec3d > rvecs, tvecs;
      // detect markers and estimate pose
      cv::aruco::detectMarkers(prime_color_mat, dictionary, corners, ids);
      for(int j = 0; j < ids.size(); j++)
      {
        if(ids[j] == 5)
        {
          averagecorners.x = 0.f;
          averagecorners.y = 0.f;
          for (int i = 0; i < corners[j].size(); i++)
          {
       		    averagecorners = averagecorners + corners[j][i];
          }
          averagecorners /= 4.0;
        }
      }

      //printf("%d\n",ids.size());
      if (ids.size() > 0)
      {
        cv::aruco::drawDetectedMarkers(displyImg, corners, ids);
        //aruco::drawDetectedMarkers(displyImg, rejected, noArray(), Scalar(100, 0, 255));
        std::vector<cv::Vec3d> rvecs,tvecs;
        cv::aruco::estimatePoseSingleMarkers(corners,0.133f,cameraMatrix_prime,distortion_prime,rvecs,tvecs);
        for(int i = 0; i<ids.size(); i++)
          {

             cv::aruco::drawAxis(displyImg,cameraMatrix_prime,distortion_prime,rvecs[i],tvecs[i],0.1);
             if (ids[i] == 0)
             {
                 if(marker_0_sum_count == 0){
                     marker_0_rvecs_sum = rvecs[i];
                     marker_0_tvecs_sum = tvecs[i];
                     marker_0_sum_count = 1;
                 }
                 Vec3d t_diff = marker_0_tvecs_sum / marker_0_sum_count - tvecs[i];
                 Vec3d r_diff = marker_0_rvecs_sum / marker_0_sum_count - rvecs[i];
                 if ((norm(t_diff) < 0.03 && norm(r_diff) < 0.1 )){
                   if(marker_0_sum_count < 100){
                     marker_0_rvecs_sum += rvecs[i];
                     marker_0_tvecs_sum += tvecs[i];
                     marker_0_sum_count ++;
                   }
                   else{
                   tvecs[i] = marker_0_tvecs_sum / marker_0_sum_count;
                   rvecs[i] = marker_0_rvecs_sum / marker_0_sum_count;
                   }
                 }
                 else{
                   marker_0_rvecs_sum = rvecs[i];
                   marker_0_tvecs_sum = tvecs[i];
                   marker_0_sum_count = 1;
                 }
             }
             //sendMarkerTF(tvecs, rvecs, ids);
             sendCameraTF(tvecs, rvecs, ids, true);

          }
      }
      //cv::aruco::drawDetectedMarkers(displyImg, rejected, noArray(), Scalar(100, 0, 255));
      ***/
      cv::resize(displyImg, displyImg, Size(960, 540));
      //displyMat_prime = displyImg.clone();
      cv::imshow("color_prime",displyImg);
      //cout << "debug" << endl;
      //waitKey(30);
      cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e)
    {
    	ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void ImageProcessor::sub_imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
      //cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
      Mat color_mat = cv_bridge::toCvShare(msg, "bgr8")->image;
      //pthread_mutex_lock(&mutex);
      sub_color_mat_queue.push_back(color_mat.clone());
      //cout << "sub queue size: " << sub_color_mat_queue.size() << "\n";
      if(sub_color_mat_queue.size() > max_queue_size){

        sub_color_mat_queue.erase(sub_color_mat_queue.begin());
      }
      //pthread_mutex_unlock(&mutex);
      Mat displyImg = sub_color_mat_queue[0].clone();

      vector<Point> joint_image_cords;
      vector<Point3f> joint_3d_cords;
      //namedWindow("Color Frame");
      vector<Point> human_image_cords;
      vector<Point3f> human_3d_cords;

      if(!sub_color_mat_queue.empty())
      {
        calculateRobotPose(joint_image_cords, joint_3d_cords, false);
        if(!joint_image_cords.empty())
        {
          //Mat depth_debug;
          //depth_debug = depth_mat.clone();
          //if(!removeRobotImage(displyImg, joint_image_cords, joint_3d_cords, depth_debug))
            //imshow("origin depth", depth_mat);
          drawRobotJoints(displyImg,joint_image_cords);
        }
        calculateHumanPose(human_image_cords, human_3d_cords, false);
        if(!human_image_cords.empty())
          draw_human_pose(displyImg, human_image_cords);
        //imshow("Color Frame", color_mat);
        if(!keypoints_sub.empty())
          drawKeyPoints(displyImg, keypoints_sub);
        drawGoalPoint(displyImg, false);
      }


/***
      cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

      Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
      //detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
      detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;
      //cv::Mat displyImg = color_mat.clone();
      //flip(displyImg,displyImg,1);
      std::vector< int > ids;
      std::vector< std::vector< cv::Point2f > > corners, rejected;

      //std::vector<cv::Point3f> world_cord;
      //  vector< Vec3d > rvecs, tvecs;
      // detect markers and estimate pose
      cv::aruco::detectMarkers(sub_color_mat, dictionary, corners, ids);
      for(int j = 0; j < ids.size(); j++)
      {
        if(ids[j] == 5)
        {
          averagecorners.x = 0.f;
          averagecorners.y = 0.f;
          for (int i = 0; i < corners[j].size(); i++)
          {
       		    averagecorners = averagecorners + corners[j][i];
          }
          averagecorners /= 4.0;
        }
      }

      //printf("%d\n",ids.size());
      if (ids.size() > 0)
      {
        cv::aruco::drawDetectedMarkers(displyImg, corners, ids);
        //aruco::drawDetectedMarkers(displyImg, rejected, noArray(), Scalar(100, 0, 255));
        std::vector<cv::Vec3d> rvecs,tvecs;
        cv::aruco::estimatePoseSingleMarkers(corners,0.133f,cameraMatrix_sub,distortion_sub,rvecs,tvecs);
        for(int i = 0; i<ids.size(); i++)
          {

             cv::aruco::drawAxis(displyImg,cameraMatrix_sub,distortion_sub, rvecs[i],tvecs[i],0.1);
             if (ids[i] == 0)
             {
                 if(marker_0_sum_count == 0){
                     marker_0_rvecs_sum = rvecs[i];
                     marker_0_tvecs_sum = tvecs[i];
                     marker_0_sum_count = 1;
                 }
                 Vec3d t_diff = marker_0_tvecs_sum / marker_0_sum_count - tvecs[i];
                 Vec3d r_diff = marker_0_rvecs_sum / marker_0_sum_count - rvecs[i];
                 if ((norm(t_diff) < 0.03 && norm(r_diff) < 0.1 )){
                   if(marker_0_sum_count < 100){
                     marker_0_rvecs_sum += rvecs[i];
                     marker_0_tvecs_sum += tvecs[i];
                     marker_0_sum_count ++;
                   }
                   else{
                   tvecs[i] = marker_0_tvecs_sum / marker_0_sum_count;
                   rvecs[i] = marker_0_rvecs_sum / marker_0_sum_count;
                   }
                 }
                 else{
                   marker_0_rvecs_sum = rvecs[i];
                   marker_0_tvecs_sum = tvecs[i];
                   marker_0_sum_count = 1;
                 }
             }
             //sendMarkerTF(tvecs, rvecs, ids);
             sendCameraTF(tvecs, rvecs, ids, false);

          }
      }
      **/
      //cv::aruco::drawDetectedMarkers(displyImg, rejected, noArray(), Scalar(100, 0, 255));
      cv::resize(displyImg, displyImg, Size(960, 540));
      //displyMat_sub = displyImg.clone();
      cv::imshow("color_sub",displyImg);
      cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e)
    {
    	ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void ImageProcessor::sendMarkerTf(vector<Point3f>& marker_position, vector<int>& ids)
{
    static tf::TransformBroadcaster marker_position_broadcaster;
    for(int i = 0; i < marker_position.size(); i++)
    {
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(marker_position[i].x, marker_position[i].y, marker_position[i].z));
        tf::Quaternion q;
        q.setRPY(0, 0, 0);
        transform.setRotation(q);
        ostringstream oss;
        oss << "marker_" << ids[i];
        marker_position_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_base", oss.str()));
    }
}

void ImageProcessor::sendMarkerTF(vector<Vec3d>& marker_trans, vector<Vec3d>& marker_rot, vector<int>& ids)
{
    Mat rot(3, 3, CV_64FC1);
    Mat rot_to_ros(3, 3, CV_64FC1);
    rot_to_ros.at<double>(0,0) = -1.0;
    rot_to_ros.at<double>(0,1) = 0.0;
    rot_to_ros.at<double>(0,2) = 0.0;
    rot_to_ros.at<double>(1,0) = 0.0;
    rot_to_ros.at<double>(1,1) = 0.0;
    rot_to_ros.at<double>(1,2) = 1.0;
    rot_to_ros.at<double>(2,0) = 0.0;
    rot_to_ros.at<double>(2,1) = 1.0;
    rot_to_ros.at<double>(2,2) = 0.0;

    static tf::TransformBroadcaster marker_position_broadcaster;
    for(int i = 0; i < ids.size(); i++)
    {
      if(ids[i] == 0 || 10)
      {

        cv::Rodrigues(marker_rot[i], rot);
        rot.convertTo(rot, CV_64FC1);


        tf::Matrix3x3 tf_rot(rot.at<double>(0,0), rot.at<double>(0,1), rot.at<double>(0,2),
                             rot.at<double>(1,0), rot.at<double>(1,1), rot.at<double>(1,2),
                             rot.at<double>(2,0), rot.at<double>(2,1), rot.at<double>(2,2));

        tf::Vector3 tf_trans(marker_trans[i][0], marker_trans[i][1], marker_trans[i][2]);
        tf::Transform transform(tf_rot, tf_trans);
        ostringstream oss;
        oss << "marker_" << ids[i];
        marker_position_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_base", oss.str()));
      }
    }
}

void ImageProcessor::sendCameraTF(vector<Vec3d>& marker_trans, vector<Vec3d>& marker_rot, vector<int>& ids, bool prime = true)
{
    Mat rot(3, 3, CV_64FC1);
    Mat rot_to_ros(3, 3, CV_64FC1);
    rot_to_ros.at<double>(0,0) = -1.0;
    rot_to_ros.at<double>(0,1) = 0.0;
    rot_to_ros.at<double>(0,2) = 0.0;
    rot_to_ros.at<double>(1,0) = 0.0;
    rot_to_ros.at<double>(1,1) = 0.0;
    rot_to_ros.at<double>(1,2) = 1.0;
    rot_to_ros.at<double>(2,0) = 0.0;
    rot_to_ros.at<double>(2,1) = 1.0;
    rot_to_ros.at<double>(2,2) = 0.0;

    string camera_frame_name("camera_base");
    if(prime)
    {
      camera_frame_name = camera_frame_name + "_1";
    }
    else
    {
      camera_frame_name = camera_frame_name + "_2";
    }

    static tf::TransformBroadcaster marker_position_broadcaster;
    for(int i = 0; i < ids.size(); i++)
    {
      if(ids[i] == 0)
      {

        cv::Rodrigues(marker_rot[i], rot);
        rot.convertTo(rot, CV_64FC1);


        tf::Matrix3x3 tf_rot(rot.at<double>(0,0), rot.at<double>(0,1), rot.at<double>(0,2),
                             rot.at<double>(1,0), rot.at<double>(1,1), rot.at<double>(1,2),
                             rot.at<double>(2,0), rot.at<double>(2,1), rot.at<double>(2,2));

        tf::Vector3 tf_trans(marker_trans[i][0], marker_trans[i][1], marker_trans[i][2]);
        tf::Transform transform(tf_rot, tf_trans);
        ostringstream oss;
        oss << "marker_" << ids[i];
        marker_position_broadcaster.sendTransform(tf::StampedTransform(transform.inverse(), ros::Time::now(), oss.str(), camera_frame_name));
      }
    }
}

void ImageProcessor::loadRobotPoseFile(string filename)
{
    ifstream inStream(filename);
    if (inStream)
    {
        vector<double> solution;
        int i = 0;
        while(!inStream.eof())
        {
            double in;
            inStream >> in;
            solution.push_back(in);
            i++;
        }
        vector<double>::iterator it = solution.end() - 1;
        solution.erase(it);
        for(int i = 0; i < solution.size(); i++)
        {
            cout << solution[i] << endl;
        }
        if(solution.size() != 10)
        {
            ROS_ERROR("Solution file invalid!");
            return;
        }
        robot_pose_tansform.setOrigin(tf::Vector3(solution[0] + solution[7], solution[1] + solution[8], solution[2] + solution[9]));
        //robot_pose_tansform.setOrigin(-tf::Vector3(solution[0], solution[1], solution[2]));
        tf::Quaternion q(solution[3], solution[4], solution[5], solution[6]);
        //q = q.inverse();
        robot_pose_tansform.setRotation(q);

        cout << "x: " << robot_pose_tansform.getRotation().x() << endl;
        cout << "y: " << robot_pose_tansform.getRotation().y() << endl;
        cout << "z: " << robot_pose_tansform.getRotation().z() << endl;
        cout << "w: " << robot_pose_tansform.getRotation().w() << endl;

    }
}
void ImageProcessor::linkToRobotTf()
{
    static tf::TransformBroadcaster robot_pose_broadcaster;
    robot_pose_broadcaster.sendTransform(tf::StampedTransform(robot_pose_tansform, ros::Time::now(), "marker_0", "base_link"));
}

bool ImageProcessor::removeRobotImage(Mat& image,vector<Point>& robot_image_pos,
                                      vector<Point3f>& robot_image_3d_pos,
                                      Mat& depth_image)
{
  /*
  vector<Point> calut_robot_image_pos;

  for (int iter = 0; iter < robot_image_pos.size() - 1; ++iter)
  {
    float Kv = (robot_image_pos[iter + 1].y - robot_image_pos[iter].y)/
               (robot_image_pos[iter + 1].x - robot_image_pos[iter].x);
    if (robot_image_pos[iter].x < robot_image_pos[iter + 1].x)
    {
      for (float i = robot_image_pos[iter].x; i < robot_image_pos[iter + 1].x; i += 3)
      {
        Point calut_robot_point(i,(int)(robot_image_pos[iter].y + Kv*(i - robot_image_pos[iter].x)));
        calut_robot_image_pos.push_back(calut_robot_point);
      }
    }
    else
    {
      for (float i = robot_image_pos[iter + 1].x; i < robot_image_pos[iter].x; i += 3)
      {
          Point calut_robot_point(i,(int)(robot_image_pos[iter].y + Kv*(i - robot_image_pos[iter + 1].x)));
          calut_robot_image_pos.push_back(calut_robot_point);
      }
    }
  }

  for (int i = 0; i < calut_robot_image_pos.size(); i++)
  {
    drawPointPiexs(image,calut_robot_image_pos,robot_image_3d_pos,depth_mat,i);
  }
*/

std::vector<Point> pending_robot_point;
std::vector<Point3f> pending_robot_3d_point;
Point robot_2d_point;
Point3f robot_3d_point;
for (int iter = 0; iter < robot_image_3d_pos.size() - 1; iter++)
{
  for (float i = 0; i <= 1.0; i += 0.1)
  {
    robot_3d_point.x = (float)(robot_image_3d_pos[iter].x + i*(robot_image_3d_pos[iter+1].x - robot_image_3d_pos[iter].x));
    robot_3d_point.y = (float)(robot_image_3d_pos[iter].y + i*(robot_image_3d_pos[iter+1].y - robot_image_3d_pos[iter].y));
    robot_3d_point.z = (float)(robot_image_3d_pos[iter].z + i*(robot_image_3d_pos[iter+1].z - robot_image_3d_pos[iter].z));
    //ROS_INFO("我估计没有问题！");
    getImageCoordinate(robot_3d_point,robot_2d_point, cameraMatrix_prime);
    pending_robot_point.push_back(robot_2d_point);
    pending_robot_3d_point.push_back(robot_3d_point);
  }
  //ROS_INFO("我估计没有问题！");
}
 //cout << "pending points size: " << pending_robot_point.size() << "\n";
int success_count = 0;
for (int i = 0; i < pending_robot_point.size(); i++)
{
  //ROS_INFO("没有错误！");
  if(drawPointPiexs(image, pending_robot_point, pending_robot_3d_point, depth_image, i))
    success_count++;
}
  if (success_count == 0)
  {
    //imshow("fail", depth_image);
    return false;
  }
  else
    return true;
}

bool ImageProcessor::drawPointPiexs(Mat& image,vector<Point>& robot_image_pos,
                                      vector<Point3f>& robot_image_3d_pos,
                                      Mat& depth_image,int numb)
{
  float depth;
  int count = 0;
  for (int i = robot_image_pos[numb].x - 30; i < robot_image_pos[numb].x + 30; i++)
  {
    for (int j = robot_image_pos[numb].y - 30; j < robot_image_pos[numb].y + 30; j++)
    {
        depth = (float)depth_image.at<uint16_t>(j, i) / 1000;
        //std::cout << depth << '\n';
        if ((depth < robot_image_3d_pos[numb].z + 0.1) && (depth > robot_image_3d_pos[numb].z - 0.1))
        {
           image.at<Vec3b>(j,i)[0] = 200;
           image.at<Vec3b>(j,i)[1] = 200;
           image.at<Vec3b>(j,i)[2] = 255;
           count++;
        }
    }
  }
  if (count > 0)
    return true;
  else
    return false;
}




int main(int argc, char** argv)
{
  ros::init(argc, argv, "display_scene");
  //cv::namedWindow("view");
  cv::startWindowThread();
  bool isCalibrationMode = true;
  bool rect = false;
  bool sendSplicedImage = false;
  string sensor = "kinect_1";
  if(argc > 1)
  {
    for(size_t i = 1; i < (size_t)argc; ++i)
    {
        printf("arg :%s\n", argv[i]);
        string arg = argv[i];
        if (arg == "false")
        {
            isCalibrationMode = false;
            ROS_INFO("calibrationMode disabled\n");
        }
        else if(arg == "rect")
        {
            rect = true;
            ROS_INFO("Subscribing to rectified image");
        }
    }
  }
  //cv::namedWindow("depth");
  cv::namedWindow("color_sub");
  cv::namedWindow("color_prime");

  ImageProcessor img_processor(rect, isCalibrationMode);
  img_processor.initCalibration();
//  ros::AsyncSpinner spinner(3);
  //use 2 threads
  ros::spin();

  return 0;

}
