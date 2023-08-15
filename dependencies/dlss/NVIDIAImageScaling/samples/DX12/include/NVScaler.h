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

#pragma once

#include <dxgi1_4.h>
#include <d3d12.h>
#include <iostream>

#include "DXUtilities.h"
#include "DeviceResources.h"
#include "Utilities.h"
#include "NIS_Config.h"
#include "d3dx12.h"

extern "C" {
    #include <dxcapi.h>
}

class NVScaler {
public:
    NVScaler(DeviceResources& deviceResources, const std::vector<std::string>& shaderPaths);
    void update(float sharpness, uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight);
    void uploadCoefficients();
    ID3D12PipelineState* getComputePSO() { return m_computePSO.Get(); }
    ID3D12Resource* getConstantBuffer() { return m_constatBuffer.Get(); }
    ID3D12RootSignature* getRootSignature() { return m_computeRootSignature.Get(); }
    std::vector<UINT> getDispatchDim() {
        return {UINT(std::ceil(m_outputWidth / float(m_blockWidth))), UINT(std::ceil(m_outputHeight / float(m_blockHeight))), 1 };
    }
    ID3D12Resource* getCoefScaler() { return m_coefScaler.Get(); }
    ID3D12Resource* getCoefUSM() { return m_coefUSM.Get(); }
protected:
    void createAlignedCoefficients(uint16_t* data, std::vector<uint16_t>& coef, uint32_t rowPitchAligned);
private:
    DeviceResources&                    m_deviceResources;
    NISConfig                           m_config;
    ComPtr<ID3D12RootSignature>         m_computeRootSignature;
    ComPtr<ID3D12PipelineState>         m_computePSO;

    ComPtr<ID3D12Resource>				m_constatBuffer;
    ComPtr<ID3D12Resource>              m_stagingBuffer;

    ComPtr<ID3D12Resource>				m_coefScaler;
    ComPtr<ID3D12Resource>				m_coefUSM;
    ComPtr<ID3D12Resource>				m_coefScalerUpload;
    ComPtr<ID3D12Resource>				m_coefUSMUpload;
    std::vector<uint16_t>               m_coefScalerHost;
    std::vector<uint16_t>               m_coefUSMHost;
    uint32_t                            m_rowPitch;

    uint32_t                            m_outputWidth;
    uint32_t                            m_outputHeight;
    uint32_t                            m_blockWidth;
    uint32_t                            m_blockHeight;
};