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
		if (k == 's')//��s����ͼƬ
		{
			imwrite(ImgName, img);
			ImgName.at(0)++;
			img.release();
			//����һ������ΪMyWindow�Ĵ���

			namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);

			//��MyWindow�Ĵ�������ʾ�洢��img�е�ͼƬ

			imshow("MyWindow", img);

			//�ȴ�ֱ���м�����

			waitKey(0);

			//����MyWindow�Ĵ���

			destroyWindow("MyWindow");
		}
		else if (k == 27)//Esc��
			break;
	}
	return 0;
}