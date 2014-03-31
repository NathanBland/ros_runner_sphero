#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <memory>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <fstream>
#include <math.h>
#define PI 3.14159265

namespace enc = sensor_msgs::image_encodings;

class Sphero {

private:
    cv::Mat frame;
    cv::NormalBayesClassifier spheroClassifier;
    cv::NormalBayesClassifier paperClassifier;
    bool fileExists(const std::string& fname);
    std::string vidSource, spheroModel, paperModel, windowName;
    double kineticScaling, imageScaling;
    ros::NodeHandle nh;
    ros::Subscriber joySub;
    ros::Publisher vel;
    ros::Publisher heading;
    image_transport::Subscriber sub;
    bool go, goPressed, publish,first;
    static void mouseCallback(int event, int x, int y, int flags, void* params);

    void doMouseCallback(int event, int x, int y, int flags, void* params){
        if(event==CV_EVENT_LBUTTONDOWN){
            if(go){
                go=false;
                ROS_INFO("Runner off");
            }else{
                go=true;
                ROS_INFO("Runner on");
            }
        }
    }

    void callbackJoyMsg(const sensor_msgs::Joy::ConstPtr& msg){
        if(msg->buttons[0] == 1 && !goPressed ){
            goPressed = true;
            if(go){
                go=false;
                ROS_INFO("Runner off");
            }else{
                go=true;
                ROS_INFO("Runner on");
            }
        }else if(goPressed && msg->buttons[0]==0){
            goPressed=false;
        }
    }
    void calibrate(const sensor_msgs::ImageConstPtr& msg){

        cv::Point2f sphero1 = detectModel(msg, true);
        double now = ros::Time::now().toSec();
        ros::Duration t(1);
        double then = t.toSec();

        while(then - now > 0){
            geometry_msgs::Twist t;
            t.linear.x = 10;
            vel.publish(t);
            now = ros::Time::now().toSec();
        }
        cv::Point2f sphero2 = detectModel(msg, true);

        double X = sqrt(pow(sphero1.x-sphero2.x,2)+(sphero1.y-sphero2.y,2));
        double Z = sqrt(pow(sphero1.x+X-sphero2.x,2)+(sphero1.y-sphero2.y,2));
        double theta = acos ((2*pow(X,2)-pow(Z,2))/(2*pow(X,2))) * 180.0 / PI;
        std_msgs::Float32 head;
        head.data = theta;
        heading.publish(head);


    }

    void spheroDetectSphero(const sensor_msgs::ImageConstPtr& msg){
        if(first){
            cv::Point2f sphero = detectModel(msg,true);
            if(sphero.x>0 && sphero.y>0){
                calibrate(msg);
                first=false;
            }
        }
        cv::Point2f paper = detectModel(msg,false);
        cv::Point2f sphero = detectModel(msg,true);
        cv::Point2f distance = cv::Point2f( sphero.x-paper.x , paper.y-sphero.y );

        if(go && sphero.x>0 && sphero.y > 0 && publish){
            geometry_msgs::Twist t;
            t.linear.x = distance.x;
            t.linear.y = distance.y;
            vel.publish(t);
            publish = false;
        } else if(!publish){
            publish = true;
        }
        imshow(windowName, frame);
        cv::waitKey(10);


    }
    cv::Point2f detectModel(const sensor_msgs::ImageConstPtr& msg, bool sphero){
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8); // extract image
        } catch(cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
        frame = cv_ptr->image;

        cv::resize(cv_ptr->image, frame, cv::Size(0,0), imageScaling,  imageScaling);
        cv::Mat vectorizedFrame = frame.reshape(1, frame.rows*frame.cols);
        cv::Mat testf, labelf;
        vectorizedFrame.convertTo(testf, CV_32FC1, 1.0/255.0, 1.0);
        if(sphero){
            spheroClassifier.predict(testf, &labelf);
        } else {
            paperClassifier.predict(testf, &labelf);
        }
        labelf = labelf.reshape(1, frame.rows);
        cv::dilate(labelf, labelf, cv::Mat::ones(3,3, CV_32FC1), cv::Point(-1,-1), 2);
        cv::erode(labelf, labelf, cv::Mat::ones(3,3, CV_32FC1), cv::Point(-1,-1), 2);
        // find the centroid of the largest blob
        cv::Mat labeli;
        labelf.convertTo(labeli, CV_8UC1);
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(labeli, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        if(contours.size() > 0){
          // look for largest blob
          size_t bigIdx = 0;
          double contourArea = 0.0;
          for(size_t ii = 0; ii < contours.size(); ii++){
            double tempA = cv::contourArea(contours.at(ii));
            if(tempA > contourArea){
              contourArea = tempA;
              bigIdx = ii;

            }
          }
          if(sphero){
              cv::drawContours(frame, contours, bigIdx, cv::Scalar(0,255,0), -1, CV_AA);
          }
          cv::Moments mu = cv::moments(contours.at(bigIdx));
          cv::Point2f mc = cv::Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );
          return mc;
        } else {
            return cv::Point2f();
            ROS_WARN("Nothing detected!!");
        }

    }


public:
    Sphero(){
        go = goPressed = publish = first= false;

        windowName = "Current Image";
        // create the display window
        cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
        cv::setMouseCallback(windowName, &Sphero::mouseCallback, this);
        ros::param::param<std::string>("~vidSource", vidSource, "/vid/video0");
        ros::param::param<double>("~kineticScaling", kineticScaling, 0.01);
        ros::param::param<double>("~imageScaling", imageScaling, 0.5);
        ros::param::param<std::string>("~spheroModel", spheroModel, "sphero.yml");
        ros::param::param<std::string>("~paperModel", paperModel, "paper.yml");

        image_transport::ImageTransport it(nh);
        sub = it.subscribe(vidSource, 1, &Sphero::spheroDetectSphero, this, image_transport::TransportHints("compressed"));
        joySub = nh.subscribe<sensor_msgs::Joy>("/joy", 1000, &Sphero::callbackJoyMsg, this);
        vel = nh.advertise<geometry_msgs::Twist>("cmd_vel", 30);
        heading = nh.advertise<std_msgs::Float32>("/set_heading",100);
        if(fileExists(spheroModel))
        {
          spheroClassifier.load(spheroModel.c_str());
          ROS_INFO("Loaded %s", spheroModel.c_str());
        } else
        {
          ROS_WARN("No sphero model loaded");
        }
        if(fileExists(paperModel))
        {
          paperClassifier.load(paperModel.c_str());
          ROS_INFO("Loaded %s", paperModel.c_str());
        } else
        {
          ROS_WARN("No paper model loaded");
        }

    }

    ~Sphero(){
      cv::destroyWindow(windowName);
    }
};


void Sphero::mouseCallback(int event, int x, int y, int flags, void* params)
{
  Sphero *self = static_cast<Sphero*>(params);
  self->doMouseCallback(event, x, y, flags, static_cast<void*>(&self->frame));
}

bool Sphero::fileExists(const std::string& fname)
{
  std::ifstream f(fname.c_str());
  if(f.good())
  {
    f.close();
    return true;
  } else
  {
    f.close();
    return false;
  }
}




int main(int argc, char **argv){
    ros::init(argc, argv, "spheroDriver");
    Sphero s;
    ros::spin();
    return(0);
}
