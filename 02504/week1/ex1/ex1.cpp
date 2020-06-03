// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
using namespace cv;
using namespace std;
int main()
{
	Mat img;
	int k;
	string ImgName = "img.jpg";
	VideoCapture cap(0);
	if (!cap.isOpened())
		return 1;
	while (1) {
		cap >> img;
		GaussianBlur(img, img, Size(3, 3), 0);
		imshow("1", img);
		k = waitKey(30);
		if (k == 's')//按s保存图片
		{
			imwrite(ImgName, img);
			ImgName.at(0)++;
			img.release();
			//创建一个名字为MyWindow的窗口

			namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);

			//在MyWindow的窗中中显示存储在img中的图片

			imshow("MyWindow", img);

			//等待直到有键按下

			waitKey(0);

			//销毁MyWindow的窗口

			destroyWindow("MyWindow");
		}
		else if (k == 27)//Esc键
			break;
	}
	return 0;
}