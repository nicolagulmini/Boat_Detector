#include <nlohmann/json.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using json = nlohmann::json;

int MAX_NEGATIVE_IMAGES = 1500;
int MAX_POSITIVE_IMAGES = 1500;

int main(int argc, char* argv[])
{
	string jsonPath, imageTmpPath, completePath, imgName, negativeImagesPath, negativeListPath;
	
	if (argc == 5)
	{
		jsonPath = argv[1]; 
		imageTmpPath = argv[2]; 
		negativeImagesPath = argv[3]; 
		negativeListPath = argv[4];
		cout << "json path: " << jsonPath << endl;
		cout << "image path: " << imageTmpPath << endl;
	}
	else
	{
		cout << "Error in json and/or images path. Return." << endl;
		exit(1);
	}
	
	// parse the json
	ifstream ifs(jsonPath);
	json jsonFile = json::parse(ifs);

	// json object with images info
	json images = jsonFile["images"];
	json annotations = jsonFile["annotations"];

	int imgId, count = 0; // default, to save and count the images later

	// take the patches from the image given the annotation info in the json
	vector<Mat> patches;
	Rect rect;

	// tmpImg, tmpImg2 are ausiliary Mat objects
	Mat img, tmpImg, tmpImg2, black, white;

	cout << "Image processing started...\n" << endl;
	for (int image_index = 0; image_index < images.size(); image_index++)
	{
		// image name to test
		imgName = images[image_index]["file_name"];
		imgId = images[image_index]["id"];
		cout << "Processing of image " << imgName << " of id " << imgId << endl;

		// complete path to open the image
		completePath = imageTmpPath + "\\" + imgName;
		img = imread(completePath);

		for (int i = 0; i < annotations.size(); i++) // first apply the filters, then (eventually) flip/crop the images because the filters are more computationally demanding
		{
			// but save ONLY patchets with cols >= 64 && rows >= 64
			if (annotations[i]["image_id"] == imgId)
			{
				vector<double> coordinates = annotations[i]["segmentation"][0];
				// since some annotations are out of image bounds...
				rect = Rect(Point(max(0.0, coordinates[0]), max(0.0, coordinates[1])), Point(min(double(img.cols), coordinates[4]), min(double(img.rows), coordinates[5])));
				if (rect.height >= 64 && rect.width >= 64)
				{
					// only blur
					tmpImg = img(rect).clone();
					GaussianBlur(tmpImg, tmpImg, Size(3, 3), 3);
					//patches.push_back(tmpImg);
					imwrite(to_string(count) + ".bmp", tmpImg);
					count++;

					/* useless. Maybe it is useful in the test step!
					// add with blur to sharpen
					tmpImg = img(rect).clone();
					tmpImg2 = tmpImg.clone();
					GaussianBlur(tmpImg, tmpImg, Size(0, 0), 2);
					addWeighted(tmpImg2, 1.5, tmpImg, -0.5, 0, tmpImg);
					patches.push_back(tmpImg);
					*/

					// additive gaussian noise
					tmpImg = img(rect).clone();
					tmpImg2 = tmpImg.clone();
					randn(tmpImg2, 128, 30);
					addWeighted(tmpImg, 0.5, tmpImg2, 0.5, 0, tmpImg);
					//patches.push_back(tmpImg);
					imwrite(to_string(count) + ".bmp", tmpImg);
					count++;

					// salt and pepper
					tmpImg = img(rect).clone();
					tmpImg2 = tmpImg.clone();
					randu(tmpImg2, 0, 255);
					black = tmpImg2 < 30;
					white = tmpImg2 > 225;
					tmpImg2 = tmpImg.clone();
					tmpImg2.setTo(255, white);
					tmpImg2.setTo(0, black);
					addWeighted(tmpImg, 0.5, tmpImg2, 0.5, 0, tmpImg);
					//patches.push_back(tmpImg);
					imwrite(to_string(count) + ".bmp", tmpImg);
					count++;

					// no change in color because the test image will be in black and white

					// prova a usare addWeighted per realizzare il cambio di brightness dell'immagine!
					// -30% of brightness
					tmpImg = img(rect).clone();
					for (int c = 0; c < tmpImg.cols; c++)
						for (int r = 0; r < tmpImg.rows; r++)
							for (int ch = 0; ch < tmpImg.channels(); ch++)
								tmpImg.at<Vec3b>(r, c)[ch] = 0.7 * tmpImg.at<Vec3b>(r, c)[ch];
					//patches.push_back(tmpImg);
					imwrite(to_string(count) + ".bmp", tmpImg);
					count++;

					// +30% of brightness
					tmpImg = img(rect).clone();
					for (int c = 0; c < tmpImg.cols; c++)
						for (int r = 0; r < tmpImg.rows; r++)
							for (int ch = 0; ch < tmpImg.channels(); ch++)
								tmpImg.at<Vec3b>(r, c)[ch] = min(255.0, 1.3 * tmpImg.at<Vec3b>(r, c)[ch]);
					//patches.push_back(tmpImg);
					imwrite(to_string(count) + ".bmp", tmpImg);
					count++;
				}
			}

			else if (annotations[i]["image_id"] > imgId) break; // stop at the right image id
		}

		/*
		for (int i = 0; i < patches.size(); i++)
		{
			imwrite(name + to_string(count) + ".bmp", patches[i]); // pay attention: permission denied!
			count++;

			// flip
			tmpImg = patches[i].clone();
			flip(tmpImg, tmpImg, 1);
			imwrite(name + to_string(count) + ".bmp", tmpImg);
			count++;

			// rotate
			tmpImg = patches[i].clone();
			rotate(tmpImg, tmpImg, ROTATE_90_CLOCKWISE);
			imwrite(name + to_string(count) + ".bmp", tmpImg);
			count++;

			// rotate
			tmpImg = patches[i].clone();
			rotate(tmpImg, tmpImg, ROTATE_90_COUNTERCLOCKWISE);
			imwrite(name + to_string(count) + ".bmp", tmpImg);
			count++;
		}
		*/

		if (count > MAX_POSITIVE_IMAGES) break;
	}
	cout << "Positive image processing completed." << endl;

	// perform the augmentation also for negative images
	cout << "Negative images processing start..." << endl;
	cout << "negative images path: " << negativeImagesPath << endl;
	cout << "negative images list path: " << negativeListPath << endl;
	count = 0;

	ifstream file(negativeListPath);
	while (getline(file, imgName) && count < MAX_NEGATIVE_IMAGES)
	{
		cout << "Processing of image " << imgName << endl;
		img = imread(imgName);

		if (!img.empty())
		{
			// blur
			tmpImg = img.clone();
			GaussianBlur(tmpImg, tmpImg, Size(3, 3), 3);
			imwrite("neg" + to_string(count) + ".bmp", tmpImg);
			count++;

			// additive gaussian noise
			tmpImg = img.clone();
			tmpImg2 = tmpImg.clone();
			randn(tmpImg2, 128, 30);
			addWeighted(tmpImg, 0.5, tmpImg2, 0.5, 0, tmpImg);
			imwrite("neg" + to_string(count) + ".bmp", tmpImg);
			count++;

			// salt and pepper
			tmpImg = img.clone();
			tmpImg2 = tmpImg.clone();
			randu(tmpImg2, 0, 255);
			black = tmpImg2 < 30;
			white = tmpImg2 > 225;
			tmpImg2 = tmpImg.clone();
			tmpImg2.setTo(255, white);
			tmpImg2.setTo(0, black);
			addWeighted(tmpImg, 0.5, tmpImg2, 0.5, 0, tmpImg);
			imwrite("neg" + to_string(count) + ".bmp", tmpImg);
			count++;

			// -30% of brightness
			tmpImg = img.clone();
			for (int c = 0; c < tmpImg.cols; c++)
				for (int r = 0; r < tmpImg.rows; r++)
					for (int ch = 0; ch < tmpImg.channels(); ch++)
						tmpImg.at<Vec3b>(r, c)[ch] = 0.7 * tmpImg.at<Vec3b>(r, c)[ch];
			imwrite("neg" + to_string(count) + ".bmp", tmpImg);
			count++;

			// +30% of brightness
			tmpImg = img.clone();
			for (int c = 0; c < tmpImg.cols; c++)
				for (int r = 0; r < tmpImg.rows; r++)
					for (int ch = 0; ch < tmpImg.channels(); ch++)
						tmpImg.at<Vec3b>(r, c)[ch] = min(255.0, 1.3 * tmpImg.at<Vec3b>(r, c)[ch]);
			imwrite("neg" + to_string(count) + ".bmp", tmpImg);
			count++;
		}
	}

	cout << "Negative image processing completed." << endl;
	exit(0);
}