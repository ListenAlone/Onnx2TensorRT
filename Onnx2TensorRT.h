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
	CUDA_ERROR = -5,//����ָ����cuda����
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
	std::string m_onnx_path;  //�����onnx�ġ��ļ���·��
	std::string m_engine_path;  //���ɳɹ����engine�ı��桾Ŀ¼��
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
	float* buffer;  //�Դ��ַ
}CLASSIFI_TEST_CLIPED_IMAGE_INFO_GPU;

typedef struct
{
	int labelIndex;
	float prob[128];
	int categoryCount;
}CLASSIFI_TEST_LABEL_RESULT;




//��Ҫ����дһ��Onnxֱ��ת��Engine����
class Onnx2TensorRT
{
public:
	Onnx2TensorRT(ONNX_Param param);
	/// <summary>
	/// onnx����tensorRT����������
	/// </summary>
	/// <param name="getInferEngine">true������������,false��ֱ�ӱ���engine�ļ�</param>
	/// <param name="engine">trueʱ���ص�����</param>
	/// <returns></returns>
	int GenerateTensorRT(bool getInferEngine = false);


	/// <summary>
	/// ������Ƶ�
	/// </summary>
	/// <param name="imageInfo">���������Դ�ͼ�������</param>
	/// <param name="imageCount">ͼ������</param>
	/// <param name="labelResult">����Ľ��</param>
	/// <param name="threshold">�������ֵ��С����ֵ����-1</param>
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

	int m_out_channel_count;  //�������
	int m_max_batch;
	int m_inputC;
	int m_inputW;
	int m_inputH;
	float m_classfiy_threshold;
	float* outputHost; //������ڴ������

	void* buffers[2];//����������Դ�ָ��
};

