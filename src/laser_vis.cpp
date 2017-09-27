#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include <ctime>

/////// classes ////////////////////////////////////////
class lpoint {
	float X, Y, Z; //world coordinates
	int x, y; // image coordinates

public:
	lpoint();
	lpoint(cv::Point3f,cv::Point2i);
	int ix() {return x;};
	int iy() {return y;};
	float wx() {return X;};
	float wy() {return Y;};
	float wz() {return Z;};
};
lpoint::lpoint(){
	X = 0;
	Y = 0;
	Z = 0;
	x = 0;
	y = 0;
}
lpoint::lpoint(cv::Point3f worldpoint, cv::Point2i imagepoint){
	X = worldpoint.x;
	Y = worldpoint.y;
	Z = worldpoint.z;
	x = imagepoint.x;
	y = imagepoint.y;
}

///// end classes ///////////////////////////////////////////

std::vector<lpoint>LaserPointsInImage;
cv::Mat K,Kinv, R_l_to_c, T_l_to_c;

/////////// functions ////////////////////////////////////////

lpoint findClosestDepthPoint(int x,int y){
	lpoint closestpoint;
	float distance, closestDistance=10000;

	for(int i = 0 ; i < LaserPointsInImage.size(); i++){
		distance = std::sqrt(std::pow((float)(LaserPointsInImage[i].ix()-x),2) + std::pow((float)(LaserPointsInImage[i].iy()-y),2));
		if (distance < closestDistance){
			closestDistance = distance;
			closestpoint = LaserPointsInImage[i];
		}
	}
	return closestpoint;
}

cv::Mat TransformLaserFrameToWorldFrame(cv::Mat pointInLaserFrame){return (R_l_to_c*pointInLaserFrame+T_l_to_c);}

cv::Point2i TransformWorldFrameToImage(cv::Mat pointInWorldFrame){
	cv::Mat image_point;
	int x_im, y_im;
	image_point = K*pointInWorldFrame;
	x_im = floor((image_point.at<float>(0)/image_point.at<float>(2))+0.5)-1;
	y_im = floor((image_point.at<float>(1)/image_point.at<float>(2))+0.5)-1;
	return cv::Point2i(x_im,y_im);
}

cv::Point3f TransformImagetoWorldFrame(cv::Point2i im,float depth){
	cv::Mat Imagepoint = (cv::Mat_<float>(3,1) << im.x*depth, im.y*depth, depth );
	cv::Mat Worldpoint = Kinv * Imagepoint;
	return cv::Point3f(Worldpoint.at<float>(0), Worldpoint.at<float>(1),Worldpoint.at<float>(2));
}

cv::Vec3b getColor(float distance){
	float min = 0;  // minimal distance
	float max = 50; //maximal distance
	int val = 5;
	int rest = 0;
	int r;
	int g;
	int b;

	if (distance<max) {
		val = std::floor(5*distance/max); // map between 0 and 1
		rest =  std::floor(((5*distance/max) - val)*255); //
	}


	switch(val)
	{
	case 0: r=255;g=rest;b=0;break;
	case 1: r=255-rest;g=255;b=0;break;
	case 2: r=0;g=255;b=rest;break;
	case 3: r=0;g=255-rest;b=255;break;
	case 4: r=rest;g=0;b=255;break;
	case 5: r=255;g=0;b=255;break;
	}

	return cv::Vec3b(r,g,b);

}


cv::Mat CreateDepthImage(std::string LaserFilename, cv::Mat K, cv::Mat R_l_to_c, cv::Mat T_l_to_c, float width, float height){
	float *px, *py, *pz;
	FILE *stream;
	int32_t num_points;
	int x_im, y_im;
	cv::Rect rectangle;
	cv::Mat depth_image, laser_point, world_point;
	std::vector<cv::Mat> laserpoints;
	cv::Point2i image_point;


	//initialze empty depth image
	depth_image=cv::Mat(height,width,CV_8UC3,cv::Scalar(0,0,0));


	// load the laser points (based on readme from the kitti data)

	// allocate 4 MB buffer (only ~130*4*4 KB are needed)
	num_points = 1000000;
	px = (float*)malloc(num_points*sizeof(float));


	// initiate pointers
	py = px+1;
	pz = px+2;

	// load file
	stream = fopen (LaserFilename.c_str(),"rb");

	//get number of points
	num_points = fread(px,sizeof(float),num_points,stream)/4;


	// loop over all the laserpoints
	for (int32_t j=0; j<num_points; j++) {
		laserpoints.push_back((cv::Mat_<float>(3,1) << *px, *py, *pz) );
		px+=4; py+=4; pz+=4; //pr+=4;
	}

	for (int32_t j=0; j<num_points; j++) {

		laser_point = laserpoints[j];

		if (laser_point.at<float>(0) > 0){
			// project to image coordinates (scale by 3rd value)
			world_point = TransformLaserFrameToWorldFrame(laser_point);
			image_point = TransformWorldFrameToImage(world_point);

			//check if it falls within the image
			if((image_point.x < width) && (image_point.x >= 0) && (image_point.y< height) && (image_point.y >= 0)){
				//past the pixel in the image
				depth_image.at<cv::Vec3b>((int)image_point.y,(int)image_point.x) = getColor(world_point.at<float>(2));

				// save the laser point
				LaserPointsInImage.push_back(lpoint(cv::Point3f(world_point.at<float>(0),world_point.at<float>(1),world_point.at<float>(2)), image_point));
			}
		}
	}


	fclose(stream);

	return depth_image;
}

int main( int argc, char** argv )
{
	//camera parameters:
	float fx = 718.856 ;
	float fy = 718.856 ;
	float cx = 607.1928;
	float cy = 185.2157;
	K = (cv::Mat_<float>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
	Kinv = K.inv();


	// set up the position differences between from laser to camera, in laser coordinates
	float xdiff = 0;
	float ydiff = 0.08;
	float zdiff = 0.27;
	R_l_to_c = (cv::Mat_<float>(3,3) << 0 , -1, 0, 0, 0, -1, 1, 0, 0  );
	T_l_to_c = (cv::Mat_<float>(3,1) << -1*xdiff, -1*ydiff, -1*zdiff );

	std::vector<cv::String> imagefilenames;
	std::vector<cv::String> laserfilenames;
	std::string path = argv[1];
	std::string imagepath = path + "/image_0";
	std::string laserpath = path + "/velodyne";
	cv::Mat image, depth_image;



	// load the kitti images
	cv::glob(imagepath, imagefilenames);
	std::cout << imagefilenames.size() <<" images found in " << imagepath << std::endl;

	// load the laser filenames
	cv::glob(laserpath, laserfilenames);
	std::cout << laserfilenames.size() <<" laserfiles found in " << laserpath << std::endl;

	//create the windows
	cv::namedWindow( "img", CV_WINDOW_AUTOSIZE);
	cv::namedWindow( "depth_img", CV_WINDOW_AUTOSIZE);
	cv::namedWindow( "Dense depth img", CV_WINDOW_AUTOSIZE);
	//cv::namedWindow( "floor_img", CV_WINDOW_AUTOSIZE);

	//loop over the images
	for (unsigned int i=0;i<imagefilenames.size();i++){
		LaserPointsInImage.clear();

		//load the image
		image = cv::imread(imagefilenames[i],CV_LOAD_IMAGE_COLOR);

		if(!image.data){
			std::cout << "loading image " << imagefilenames[i] << " failed, skipping ..." << std::endl;
			//continue;
		}

		//create the depth image
		depth_image = CreateDepthImage(laserfilenames[i], K, R_l_to_c, T_l_to_c, image.size().width, image.size().height);


		///////////////////////////////// BEING ADDED, CREATE DENSE DEPTHMAP ////////////////////////////////////////////////
		std::time_t starttime = std::time(0);
		cv::Mat DenseDepthImage = cv::Mat(376,1241,CV_8UC3,cv::Scalar(0,0,0));
		float depth;
		lpoint laserpoint;
		for (int x = 0; x<image.size().width-1;x++){
			for (int y = 0; y<image.size().height-1; y++){
				laserpoint = findClosestDepthPoint(x,y);
				depth = laserpoint.wz();
				DenseDepthImage.at<cv::Vec3b>(y,x) = getColor(depth);
			}
		}
		std::time_t runtime = std::time(0)-starttime;
		std::cout << "total time to create this dense depth image is: " << runtime << "seconds" << std::endl;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////// BEING ADDED, STIXELS REPRESENTATION, CREATING FLOOR IMAGE /////////////////////////////////////////////////////
//		int laserHeightInImage = (image.size().height/2);
//		float depth, cameraheight = 1.65, heightTreshold = 0.1;
//		cv::Mat floorImage = cv::Mat(376,1241,CV_8UC3,cv::Scalar(0,0,0));
//		cv::Point3f Worldpoint;
//
//
//		// loop from bottom to top over each col
//		for(int x=0;x<image.size().width-1;x++){
//			for(int y =image.size().height-1 ; y> laserHeightInImage; y--){
//				//std::cout << "x: "<<x <<"y: " << y << std::endl;
//
//				// get depth from closest depth point
//				depth = findClosestDepthPoint(x,y).wz();
//
//				//transfer to world point
//				Worldpoint = TransformImagetoWorldFrame(cv::Point2i(x,y),depth);
//
//
//				if (Worldpoint.y >= cameraheight-heightTreshold && Worldpoint.y <= cameraheight+heightTreshold){
//
//
//					floorImage.at<cv::Vec3b>(y,x) = cv::Vec3b(0,255,0);
//
//				}
//			}
//		}



		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		// show the images

		while(true){
			imshow("img", image);
			imshow("depth_img", depth_image);
			//imshow("floor_img", floorImage);
			imshow("Dense depth img", DenseDepthImage);
			int k = cv::waitKey(1);
			if (k == 27 & 0xFF){
				cv::destroyAllWindows();
				break;
			}
		}

	}

	return 1;
}
