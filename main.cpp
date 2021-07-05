#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>

using namespace cv;
using namespace std;
using json = nlohmann::json;

string FileName(string str)
{
	size_t found = str.find_last_of("/\\");
	string name = str.substr(found + 1); 
	return name;
}

int allZero(vector<double> values)
{
	for (int i = 0; i < values.size(); i++)
		if (values[i] != 0)
			return 0;
	return 1;
}

int main(int argc, char* argv[])
{
	cout << "Welcome to the Boat Detection program. Loading..." << endl;

	if (argc < 3)
	{
		cout << "Error in command line parameters. \nPlease pass an image path as first parameter and the cascade path as second parameter. If there is a ground truth, provide a json as third argument. Exit." << endl;
		exit(1);
	}

	Mat img, originalImg;
	string imgPath = argv[1]; 
	string cascade = argv[2];

	originalImg = imread(imgPath);
	if (originalImg.empty() || originalImg.cols < 64 || originalImg.rows < 64)
	{
		cout << endl << "Error. Image empty or too small. Please take an image with minimum size 64x64. Exit." << endl; exit(1);
	}

	// image processing
	cvtColor(originalImg, img, COLOR_BGR2GRAY);

	// resizing image to perform a faster and a more accurate computation
	// since detectMultiScale produces squares, the image should be a square too. 
	// Then I will perform another resize back to the original dimension, resizing also the boxes.
	int newDim = 300;
	resize(img, img, Size(newDim, newDim), 0, 0, INTER_LANCZOS4);
	
	/*
		With different types of interpolation I obtain very different results! Look for the best interpolation.
		INTER_NEAREST
		INTER_LINEAR 
		INTER_AREA 
		INTER_CUBIC 
		INTER_LANCZOS4
	*/
	
	equalizeHist(img, img);

	vector<Rect> rectangles;
	vector<double> weights;
	vector<int> rejLevels;
	CascadeClassifier cascadeClassifier;
	
	if (!cascadeClassifier.load(cascade))
	{
		cout << "Error loading the cascade. Exit." << endl; exit(1);
	}
	
	cascadeClassifier.detectMultiScale(img, rectangles, rejLevels, weights, 1.05, 6, 0, Size(64, 64), Size(), true);
	int detected = rectangles.size();
	if (detected == 0) cout << "No boats detected." << endl;
	else
	{
		int originalRectW, originalRectH;
		double maxWeight = *max_element(weights.begin(), weights.end());
		double threshold = 0.6;

		cout << "\nNumber of detected boats: " << detected << endl;
		int count = 0; // to count the shown boxes
		vector<Rect> resizedRectangles;
		for (int i = 0; i < detected; i++)
		{
			if (weights[i] >= threshold * maxWeight) 
			{
				// perform the conversion to show the rectangles in the original image
				Point originalRectPoint(rectangles[i].x * originalImg.cols / newDim, rectangles[i].y * originalImg.rows / newDim);
				originalRectW = rectangles[i].width * originalImg.cols / newDim;
				originalRectH = rectangles[i].height * originalImg.rows / newDim;
				Rect originalRect(originalRectPoint.x, originalRectPoint.y, originalRectW, originalRectH);
				resizedRectangles.push_back(originalRect);
				rectangle(originalImg, originalRect, Scalar(0, 0, 255), 8);
				cout << "The certainty of the detection " << originalRect << " is: " << weights[i] << endl;

				count++;
			}
		}
		if (count < detected) cout << "Shown " << count << " boxes because of the low certainty of the unshown detections." << endl;
		
		namedWindow("Detected boats", WINDOW_NORMAL);
		imshow("Detected boats", originalImg);
		waitKey(0);

		// performance computation: if there are more boats, return the best IoU for each detection
		if (argc == 4)
		{
			string jsonGroundTruthPath = argv[3];
			ifstream ifs(jsonGroundTruthPath);
			json jsonFile = json::parse(ifs);
			string imgName = FileName(imgPath);
			int imgId = -1;
			json tmp = jsonFile["images"];
			for (int i = 0; i < tmp.size(); i++) // take the image id given the image name
			{
				if (tmp[i]["file_name"] == imgName)
				{
					imgId = tmp[i]["id"];
					break;
				}
			}

			if (imgId == -1) // check if there is the image
			{
				cout << "Image not found in the provided json." << endl; exit(1);
			}

			tmp = jsonFile["annotations"];
			vector<Rect> groundTruthRectangles;
			for (int i = 0; i < tmp.size(); i++)
			{
				if (tmp[i]["image_id"] == imgId)
				{
					json bbox = tmp[i]["bbox"];
					groundTruthRectangles.push_back(Rect(bbox[0], bbox[1], bbox[2], bbox[3]));
				}
				else if (tmp[i]["image_id"] > imgId) break;
			}
			
			cout << groundTruthRectangles.size() << " ground truth annotations found:" << endl;
			for (int i = 0; i < groundTruthRectangles.size(); i++)
			{
				cout << groundTruthRectangles[i] << endl;
				rectangle(originalImg, groundTruthRectangles[i], Scalar(0, 255, 0), 8);
			}

			namedWindow("Ground Truth", WINDOW_NORMAL);
			imshow("Ground Truth", originalImg);
			waitKey(0);

			cout << "Measuring performances..." << endl;
			Rect intersection, unionRect;
			double intersectionOverUnion;
			vector<double> iouValues;
			for (int i = 0; i < resizedRectangles.size(); i++)
				for (int j = 0; j < groundTruthRectangles.size(); j++)
				{
					intersection = resizedRectangles[i] & groundTruthRectangles[j];
					unionRect = resizedRectangles[i] | groundTruthRectangles[j];
					intersectionOverUnion = (double)(intersection.width * intersection.height) / (unionRect.width * unionRect.height);
					iouValues.push_back(intersectionOverUnion);
					cout << "Intersection Over Union between detected " << resizedRectangles[i] << " and truth " << groundTruthRectangles[j] << ": " << intersectionOverUnion << endl;
				}	
			if (allZero(iouValues) == 1) cout << "What a bad detection..." << endl;
		}
	}

	exit(0);
}