#pragma once
#include <iostream>
#include <vector>
#include "NvInfer.h"

enum xErrorCode
{
	SUCCESS = 0,
	N0_FIND_CUDA = -1,
	CREATE_INFER_BUILDER_ERROR = -2,
	CUDA_VERSION_LOW_ERROR = -3,
	FILE_OR_DIR_NOT_FIND = -4,
	CUDA_ERROR = -5,//不好指明的cuda错误
	NETWORK_DEFINITION_ERROR = -6,
	BUILDER_CONFIG_ERROR = -7,
	PARSER_ONNX_FAILED = -8,
	CREATE_CUDA_ENGINE_FAILED = -9,
	CREATE_EXECUTE_CONTEXT_FAIL = -10,
	WRITE_ENGINE_ERROR = -11,
	CONTEXT_NOT_FIND = -12,
	INPUT_IMG_SIZE_ERROR = -13,
	INPUT_IMG_BATCH_ERROR = -14,
	ENQUEUEV2_ERROR = -15,
};


class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg) override
	{
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
};



enum RUN_MODE
{
	FP32 = 0,
	FP16 = 1,
	INT8 = 2
};




struct ONNX_Param
{
	std::string m_onnx_path;  //传入的onnx的【文件】路径
	std::string m_engine_path;  //生成成功后的engine的保存【目录】
	int m_deviceId = 0;
	int batch_size;
	RUN_MODE run_mode = FP32;
	std::vector<std::string> inputTensorNames; 
	std::vector<std::string> outputTensorNames; 
};


typedef struct
{
	int width;
	int height;
	int channel;
	float* buffer;  //显存地址
}CLASSIFI_TEST_CLIPED_IMAGE_INFO_GPU;

typedef struct
{
	int labelIndex;
	float prob[128];
	int categoryCount;
}CLASSIFI_TEST_LABEL_RESULT;




//主要是想写一个Onnx直接转成Engine的类
class Onnx2TensorRT
{
public:
	Onnx2TensorRT(ONNX_Param param);
	/// <summary>
	/// onnx生成tensorRT的推理引擎
	/// </summary>
	/// <param name="getInferEngine">true返回推理引擎,false就直接保存engine文件</param>
	/// <param name="engine">true时返回的引擎</param>
	/// <returns></returns>
	int GenerateTensorRT(bool getInferEngine = false);


	/// <summary>
	/// 分类的推导
	/// </summary>
	/// <param name="imageInfo">传进来的显存图像的数据</param>
	/// <param name="imageCount">图像张数</param>
	/// <param name="labelResult">分类的结果</param>
	/// <param name="threshold">分类的阈值，小于阈值返回-1</param>
	/// <returns></returns>
	int Do_Infer_Classify(CLASSIFI_TEST_CLIPED_IMAGE_INFO_GPU* imageInfo, int imageCount, 
		CLASSIFI_TEST_LABEL_RESULT* labelResult, float threshold);

	~Onnx2TensorRT();


private:
	bool CheckFileOrDirIsExist(std::string file_name);
	xErrorCode GetEngineUniqueName(std::string& name);


private:
	ONNX_Param m_param;
	Logger gLogger;
	nvinfer1::IExecutionContext* m_context;
	nvinfer1::ICudaEngine* m_engine;
	std::vector<nvinfer1::Dims> m_inputDims;
	std::vector<nvinfer1::Dims> m_outputDims;
	cudaStream_t m_cudaStreamProcess;
	int m_inputIndex;
	int m_outputIndex;

	int m_out_channel_count;  //类别数量
	int m_max_batch;
	int m_inputC;
	int m_inputW;
	int m_inputH;
	float m_classfiy_threshold;
	float* outputHost; //输出的内存的数据

	void* buffers[2];//输入输出的显存指针
};

