#include <iostream>
#include "Onnx2TensorRT.h"
#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "opencv2/dnn/dnn.hpp"

using namespace std;
using namespace cv;

void Test_Onnx()
{
	ONNX_Param param;
	param.batch_size = 200;
	param.m_deviceId = 0;
	param.m_engine_path = R"(D:\networksTest\class\default)";
	param.m_onnx_path = R"(D:\networksTest\class\default\ClassiftyForResNet18.onnx)";
	param.run_mode = FP16;

	Onnx2TensorRT rt(param);
	int rtn = rt.GenerateTensorRT(true);
	if (rtn != 0)
		cout << rtn << endl;

	CLASSIFI_TEST_CLIPED_IMAGE_INFO_GPU* imageInfo = new CLASSIFI_TEST_CLIPED_IMAGE_INFO_GPU[param.batch_size];
	CLASSIFI_TEST_LABEL_RESULT* labelResult = new CLASSIFI_TEST_LABEL_RESULT[param.batch_size];
	float threshold = 0.25f;

	Mat image = imread(R"(D:\networksTest\class\default\1.jpg)");

	//我的网络是64*64*3的
	Mat floatMat = cv::dnn::blobFromImage(image, 1 / 255., Size(64, 64), Scalar(), true);

	float* m_floatPtr;
	cudaMalloc(&m_floatPtr, 64 * 64 * 3 * sizeof(float));
	cudaMemcpy(m_floatPtr, floatMat.data, 64 * 64 * 3 * sizeof(float), cudaMemcpyHostToDevice);


	for (int i = 0; i < param.batch_size; ++i)
	{
		imageInfo[i].width = 64;
		imageInfo[i].height = 64;
		imageInfo[i].channel = 3;
		cudaMalloc(&imageInfo[i].buffer, 64 * 64 * 3 * sizeof(float));
		cudaMemcpy(imageInfo[i].buffer, m_floatPtr, 64 * 64 * 3 * sizeof(float),cudaMemcpyDeviceToDevice);
	}
	



	rtn = rt.Do_Infer_Classify(imageInfo, param.batch_size, labelResult, threshold);
	if (rtn != 0)
		cout << rtn << endl;


	for (int i = 0; i < param.batch_size; ++i)
	{
		int index = labelResult[i].labelIndex;
		cout << index << " " << labelResult[i].prob[index] << endl;
	}

	for (int i = 0; i < param.batch_size; ++i)
	{
		cudaFree(imageInfo[i].buffer);
	}

	delete[] imageInfo;
	delete[] labelResult;
	cudaFree(m_floatPtr);
}


int main()
{
	Test_Onnx();
	return 0;
}