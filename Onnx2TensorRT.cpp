#include "Onnx2TensorRT.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <io.h>

#pragma warning(disable : 4996)

using namespace nvinfer1;
using namespace std;
using namespace nvonnxparser;

Logger gLogger;

Onnx2TensorRT::Onnx2TensorRT(ONNX_Param param)
	:m_param(param)
{
	if (m_context) m_context->destroy();
	m_context = nullptr;

	if (m_engine) m_engine->destroy();
	m_engine = nullptr;
}

int Onnx2TensorRT::GenerateTensorRT(bool getInferEngine)
{
	xErrorCode err = SUCCESS;
	
	//检查onnx文件是否存在
	bool file_status = CheckFileOrDirIsExist(m_param.m_onnx_path);
	if (!file_status) return FILE_OR_DIR_NOT_FIND;

	//检查存储目录是否存在
	file_status = CheckFileOrDirIsExist(m_param.m_engine_path);
	if (!file_status) return FILE_OR_DIR_NOT_FIND;

	//检查engine是否存在
	//设置GPU
	cudaError_t cudaErr = cudaSetDevice(m_param.m_deviceId);
	if (cudaErr != cudaSuccess) return N0_FIND_CUDA;

	//①生成engine的唯一文件名。engine和cuda版本有关
	string engine_name;
	err = GetEngineUniqueName(engine_name);
	if (err != SUCCESS) return err;

	//②判断engine文件是否存在
	string engine_all_path = m_param.m_engine_path + "\\" + engine_name;
	file_status = CheckFileOrDirIsExist(engine_all_path);


	cudaError_t error = cudaStreamCreate(&m_cudaStreamProcess);
	if (m_engine)
		m_engine->destroy();
	m_engine = nullptr;




	//已经存在就直接返回
	if (file_status)
	{
		//如果不需要获取就直接返回
		if (!getInferEngine) return SUCCESS;
		
		//需要则反序列化
		IRuntime* runtime = createInferRuntime(gLogger);

		FILE* pFile = fopen(engine_all_path.c_str(), "rb");
		if (!pFile) return WRITE_ENGINE_ERROR;
		fseek(pFile, 0, SEEK_END);
		long fileSize = ftell(pFile);
		char* buffer = new char[fileSize];
		fseek(pFile, 0, SEEK_SET);
		size_t result = fread(buffer, 1, fileSize, pFile);
		if (result != fileSize) {
			delete[] buffer;
			return WRITE_ENGINE_ERROR;
		}
		fclose(pFile);

		m_engine = runtime->deserializeCudaEngine(buffer, fileSize);
		delete[] buffer;
		if(!m_engine) return CREATE_CUDA_ENGINE_FAILED;
	}
	else
	{

		//====================engine不存在,准备生成========================

		//创建build对象
		IBuilder* builder = createInferBuilder(gLogger);
		if (!builder) return CREATE_INFER_BUILDER_ERROR;
		builder->setMaxBatchSize(m_param.batch_size);

		//onnx解析器不支持不确定batch 所以设置成明确batch
		const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
		if (!network) return NETWORK_DEFINITION_ERROR;


		//onnx文件的解析器 和 network相关联
		IParser* parser = nvonnxparser::createParser(*network, gLogger);
		//通过解析器器将onnx文件加载到network中
		bool parser_status = parser->parseFromFile(m_param.m_onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
		if (!parser_status) return PARSER_ONNX_FAILED;


		//创建build的一些参数设置
		IBuilderConfig* config = builder->createBuilderConfig();
		if (!config) return BUILDER_CONFIG_ERROR;

		config->setMaxWorkspaceSize(16 * (1 << 20));
		switch (m_param.run_mode)
		{
		case FP16:
			config->setFlag(BuilderFlag::kFP16);
			break;
		case INT8:
			config->setFlag(BuilderFlag::kINT8);
			break;
		}

		//获取第一层的name
		Dims dim = network->getInput(0)->getDimensions();
		if (dim.d[0] == -1)  //-1表示是动态网络
		{
			const char* name = network->getInput(0)->getName();
			IOptimizationProfile* profile = builder->createOptimizationProfile();
			profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
			profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(m_param.batch_size, dim.d[1], dim.d[2], dim.d[3]));
			profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(m_param.batch_size, dim.d[1], dim.d[2], dim.d[3]));
			config->addOptimizationProfile(profile);
		}


		//通过network和额外的参数,生成Engine
		m_engine = builder->buildEngineWithConfig(*network, *config);
		if (!m_engine) return CREATE_CUDA_ENGINE_FAILED;

		//序列化操作
		IHostMemory* gieModelStream = m_engine->serialize();
		FILE* pFile = fopen(engine_all_path.c_str(), "wb");
		if (!pFile) return WRITE_ENGINE_ERROR;

		size_t wirtedCount = fwrite(gieModelStream->data(), 1, gieModelStream->size(), pFile);
		if (wirtedCount != gieModelStream->size()) return WRITE_ENGINE_ERROR;
		fclose(pFile);


		//销毁操作
		gieModelStream->destroy();
		config->destroy();
		builder->destroy();
		network->destroy();
		parser->destroy();
	}



	if (getInferEngine)
	{
		m_context = m_engine->createExecutionContext();
		if (!m_context) return CREATE_EXECUTE_CONTEXT_FAIL;
	}

	m_param.inputTensorNames.clear();
	m_param.outputTensorNames.clear();
	int bindings = m_engine->getNbBindings();
	for (int i = 0; i < bindings; i++)
	{
		string name = m_engine->getBindingName(i);
		bool isInput = m_engine->bindingIsInput(i);
		std::cout << name << " is Input: " << isInput << endl;
		if (isInput) {
			m_param.inputTensorNames.push_back(name);
		}
		else {
			m_param.outputTensorNames.push_back(name);
		}
	}

	for (int i = 0; i < m_param.inputTensorNames.size(); i++) {
		m_inputIndex = m_engine->getBindingIndex(m_param.inputTensorNames[i].c_str());
		m_inputDims.push_back(m_engine->getBindingDimensions(m_inputIndex));
	}

	for (int i = 0; i < m_param.outputTensorNames.size(); i++) {
		m_outputIndex = m_engine->getBindingIndex(m_param.outputTensorNames[i].c_str());
		m_outputDims.push_back(m_engine->getBindingDimensions(m_outputIndex));
	}

	//销毁操作

	m_out_channel_count = m_outputDims[0].d[1];
	int inputDims = m_inputDims[0].nbDims;
	if (m_inputDims[0].d[inputDims - 4] != -1)
	{
		m_max_batch = m_inputDims[0].d[inputDims - 4];
	}
	else
	{
		m_max_batch = m_param.batch_size;
	}
	m_inputC = m_inputDims[0].d[inputDims - 3];
	m_inputW = m_inputDims[0].d[inputDims - 2];
	m_inputH = m_inputDims[0].d[inputDims - 1];

	if(cudaSuccess != cudaMalloc(&buffers[0], m_max_batch * m_inputC * m_inputW * m_inputH * sizeof(float)))
		return CUDA_ERROR;
	if(cudaSuccess != cudaMalloc(&buffers[1], m_max_batch * m_out_channel_count * sizeof(float)))
		return CUDA_ERROR;

	outputHost = new float[m_max_batch * m_out_channel_count]{0};

	return SUCCESS;
}

int Onnx2TensorRT::Do_Infer_Classify(CLASSIFI_TEST_CLIPED_IMAGE_INFO_GPU* imageInfo, int imageCount, CLASSIFI_TEST_LABEL_RESULT* labelResult, float threshold)
{
	if (!m_context || !m_engine) return CONTEXT_NOT_FIND;

	if (imageCount > m_max_batch)
		return	INPUT_IMG_BATCH_ERROR;

	m_classfiy_threshold = threshold;

	//分类数目
	int orgin_size = m_inputC * m_inputH * m_inputW;
	int orgin_zoom_size = orgin_size * sizeof(float); //每个输入图像的实际大小

	if (m_inputC != imageInfo[0].channel || m_inputW != imageInfo[0].width || m_inputH != imageInfo[0].height)
		return	INPUT_IMG_SIZE_ERROR;

	for (int i = 0; i < imageCount; ++i) {
		cudaError_t error = cudaMemcpyAsync((float*)(buffers[m_inputIndex]) + i * orgin_size, imageInfo[i].buffer,
			orgin_zoom_size, cudaMemcpyDeviceToDevice, m_cudaStreamProcess);
	}

	if (m_inputDims[0].d[0] == -1)//-1表示是动态网络，需要设置输入的大小
	{
		m_context->setBindingDimensions(0, Dims4(m_max_batch, m_inputC, m_inputH, m_inputW));
	}

	if (!m_context->enqueueV2(buffers, m_cudaStreamProcess, nullptr))
		return ENQUEUEV2_ERROR;

	cudaMemcpyAsync(outputHost, buffers[m_outputIndex], m_max_batch * m_out_channel_count * sizeof(float), cudaMemcpyDeviceToHost, m_cudaStreamProcess);
	cudaStreamSynchronize(m_cudaStreamProcess);


	int sizeOut = 1 * 1 * m_out_channel_count;
	for (int i = 0; i < imageCount; i++) {
		int label = 0;
		float maxProb = 0;
		for (int j = 0; j < m_out_channel_count; j++) {
			float prob = *(outputHost + i * sizeOut + j);
			if (prob > maxProb) {
				label = j;
				maxProb = prob;
			}
			labelResult[i].prob[j] = prob;
		}

		if (maxProb > m_classfiy_threshold) {
			labelResult[i].labelIndex = label;
		}
		else {
			labelResult[i].labelIndex = -1;
		}
		labelResult[i].categoryCount = m_out_channel_count;

	}

	return SUCCESS;
}

Onnx2TensorRT::~Onnx2TensorRT()
{
	if (buffers[0])
	{
		cudaFree(buffers[0]);
		buffers[0] = nullptr;
	}
	if (buffers[1])
	{
		cudaFree(buffers[1]);
		buffers[1] = nullptr;
	}

	if (outputHost)
	{
		delete[] outputHost;
		outputHost = nullptr;
	}
	m_context->destroy();
	m_context = nullptr;

	m_engine->destroy();
	m_engine = nullptr;
	
	if (m_cudaStreamProcess)
		cudaStreamDestroy(m_cudaStreamProcess);

}

bool Onnx2TensorRT::CheckFileOrDirIsExist(std::string file_name)
{
	return (_access(file_name.c_str(), 0) == 0);
}

xErrorCode Onnx2TensorRT::GetEngineUniqueName(std::string& name)
{
	cudaDeviceProp prop;
	if (cudaSuccess == cudaGetDeviceProperties(&prop, m_param.m_deviceId)) {
		if (prop.major >= 1) {
			string deviceName = prop.name;
			int version = 0;
			if (cudaSuccess == cudaRuntimeGetVersion(&version))
			{
				//获取onnx的文件名
				int len1 = m_param.m_onnx_path.find_last_of('\\') + 1;
				int len2 = m_param.m_onnx_path.find_last_of(".onnx") + 1;
				string onnx_file_name = m_param.m_onnx_path.substr(len1, len2 - strlen(".onnx") - len1);
				
				switch (m_param.run_mode)
				{
				case FP16:
					onnx_file_name+="_FP16";
					break;
				case INT8:
					onnx_file_name += "_INT8";
					break;
				default:
					onnx_file_name += "_FP32";
					break;
				}
				name = onnx_file_name + "_" + deviceName + "_cuda" + to_string(version) + ".engine";
				return SUCCESS;
			}
			else
			{
				return CUDA_ERROR;
			}
		}
		else {
			return CUDA_VERSION_LOW_ERROR;
		}
	}
	else
	{
		return CUDA_ERROR;
	}
}
