// The MIT License(MIT)
//
// Copyright(c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files(the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "NVSharpen.h"

#include <iostream>
#include "DXUtilities.h"
#include "DeviceResources.h"
#include "Utilities.h"


NVSharpen::NVSharpen(DeviceResources& deviceResources,  const std::vector<std::string>& shaderPaths)
    : m_deviceResources(deviceResources)
    , m_outputWidth(1)
    , m_outputHeight(1)
{
    std::string shaderName = "NIS_Main.hlsl";
    std::string shaderPath;
    for (auto& e : shaderPaths)
    {
        if (std::filesystem::exists(e + "/" + shaderName))
        {
            shaderPath = e + "/" + shaderName;
            break;
        }
    }
    if (shaderPath.empty())
        throw std::runtime_error("Shader file not found" + shaderName);

    NISOptimizer opt(false, NISGPUArchitecture::NVIDIA_Generic);
    m_blockWidth = opt.GetOptimalBlockWidth();
    m_blockHeight = opt.GetOptimalBlockHeight();
    uint32_t threadGroupSize = opt.GetOptimalThreadGroupSize();

    std::wstring wNIS_BLOCK_WIDTH = widen(toStr(m_blockWidth));
    std::wstring wNIS_BLOCK_HEIGHT = widen(toStr(m_blockHeight));
    std::wstring wNIS_THREAD_GROUP_SIZE = widen(toStr(threadGroupSize));
    std::wstring wNIS_HDR_MODE = widen(toStr(uint32_t(NISHDRMode::None)));
    std::vector<DxcDefine> defines {
        {L"NIS_SCALER", L"0"},
        {L"NIS_HDR_MODE", wNIS_HDR_MODE.c_str()},
        {L"NIS_USE_HALF_PRECISION", L"1"},
        {L"NIS_HLSL_6_2", L"1"},
        {L"NIS_BLOCK_WIDTH", wNIS_BLOCK_WIDTH.c_str()},
        {L"NIS_BLOCK_HEIGHT", wNIS_BLOCK_WIDTH.c_str()},
        {L"NIS_THREAD_GROUP_SIZE", wNIS_BLOCK_HEIGHT.c_str()},
    };

    ComPtr<IDxcLibrary> library;
    DX::ThrowIfFailed(DxcCreateInstance(CLSID_DxcLibrary, __uuidof(IDxcLibrary), &library));
    ComPtr<IDxcCompiler> compiler;
    DX::ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, __uuidof(IDxcCompiler), &compiler));
    std::wstring wShaderFilename = widen(shaderPath);

    uint32_t codePage = CP_UTF8;
    ComPtr<IDxcBlobEncoding> sourceBlob;
    DX::ThrowIfFailed(library->CreateBlobFromFile(wShaderFilename.c_str(), &codePage, &sourceBlob));

    ComPtr<IDxcIncludeHandler> includeHandler;
    library->CreateIncludeHandler(&includeHandler);
    std::vector<LPCWSTR> args{ L"-O3", L"-enable-16bit-types" };
    ComPtr<IDxcOperationResult> result;
    HRESULT hr = compiler->Compile(sourceBlob.Get(), wShaderFilename.c_str(), L"main", L"cs_6_2", args.data(), uint32_t(args.size()),
        defines.data(), uint32_t(defines.size()), includeHandler.Get(), &result);
    if (SUCCEEDED(hr))
        result->GetStatus(&hr);
    if (FAILED(hr))
    {
        if (result)
        {
            ComPtr<IDxcBlobEncoding> errorsBlob;
            hr = result->GetErrorBuffer(&errorsBlob);
            if (SUCCEEDED(hr) && errorsBlob)
            {
                wprintf(L"Compilation failed with errors:\n%hs\n", (const char*)errorsBlob->GetBufferPointer());
            }
        }
        DX::ThrowIfFailed(hr);
    }
    ComPtr<IDxcBlob> computeShaderBlob;
    result->GetResult(&computeShaderBlob);

    m_deviceResources.CreateBuffer(sizeof(NISConfig), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, &m_stagingBuffer);
    m_deviceResources.CreateBuffer(sizeof(NISConfig), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &m_constatBuffer);

    constexpr uint32_t nParams = 4;
    CD3DX12_DESCRIPTOR_RANGE descriptorRange[nParams] = {};
    descriptorRange[0] = CD3DX12_DESCRIPTOR_RANGE(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
    descriptorRange[1] = CD3DX12_DESCRIPTOR_RANGE(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);
    descriptorRange[2] = CD3DX12_DESCRIPTOR_RANGE(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    descriptorRange[3] = CD3DX12_DESCRIPTOR_RANGE(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
    CD3DX12_ROOT_PARAMETER m_rootParams[nParams] = {};
    m_rootParams[0].InitAsDescriptorTable(1, &descriptorRange[0]);
    m_rootParams[1].InitAsDescriptorTable(1, &descriptorRange[1]);
    m_rootParams[2].InitAsDescriptorTable(1, &descriptorRange[2]);
    m_rootParams[3].InitAsDescriptorTable(1, &descriptorRange[3]);

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.NumParameters = nParams;
    rootSignatureDesc.pParameters = m_rootParams;
    rootSignatureDesc.NumStaticSamplers = 0;
    rootSignatureDesc.pStaticSamplers = nullptr;
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serializedSignature;
    DX::ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serializedSignature, nullptr));

    // Create the root signature
    DX::ThrowIfFailed(m_deviceResources.device()->CreateRootSignature(0, serializedSignature->GetBufferPointer(),serializedSignature->GetBufferSize(),
            __uuidof(ID3D12RootSignature),&m_computeRootSignature));
    m_computeRootSignature->SetName(L"NVSharpen");
    // Create compute pipeline state
    D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
    descComputePSO.pRootSignature = m_computeRootSignature.Get();
    descComputePSO.CS.pShaderBytecode = computeShaderBlob->GetBufferPointer();
    descComputePSO.CS.BytecodeLength = computeShaderBlob->GetBufferSize();

    DX::ThrowIfFailed(m_deviceResources.device()->CreateComputePipelineState(&descComputePSO, __uuidof(ID3D12PipelineState), &m_computePSO));
    m_computePSO->SetName(L"NVSharpen Compute PSO");
}

void NVSharpen::update(float sharpness, uint32_t inputWidth, uint32_t inputHeight)
{
    NVSharpenUpdateConfig(m_config, sharpness,
        0, 0, inputWidth, inputHeight, inputWidth, inputHeight,
        0, 0, NISHDRMode::None);
    m_deviceResources.UploadBufferData(&m_config, sizeof(NISConfig), m_constatBuffer.Get(), m_stagingBuffer.Get());
    m_outputWidth = inputWidth;
    m_outputHeight = inputHeight;
}