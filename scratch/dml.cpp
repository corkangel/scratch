#include "stensor.h"

#define NOMINMAX

#include <wil/result.h>
#include <wil/resource.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <optional>

#pragma warning(disable : 4238) // References to temporary classes are okay because they are only used as function parameters.

#include "d3dx12.h" // The D3D12 Helper Library that you downloaded.
#include <dxgi1_4.h>
#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS


#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h> // The DirectML header from the Windows SDK.
#include "scratch/DirectMLX.h"  // open source utils for DirectML


#define USE_DMLX 1

// Enable this to show element-wise identity multiplied by a scalar placed in a constant node.
// This only applies to the DirectMLX API that builds a graph.
#define MULTIPLY_WITH_SCALAR_CONSTANT 0

using Microsoft::WRL::ComPtr;


struct DmlAppGlobals
{
    ComPtr<ID3D12Device> d3D12Device;
    ComPtr<ID3D12CommandQueue> commandQueue;
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    ComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;
    ComPtr<IDMLBindingTable> dmlBindingTable;
    ComPtr<IDMLCompiledOperator> dmlCompiledOperator;
    ComPtr<IDMLCommandRecorder> dmlCommandRecorder;

    ComPtr<ID3D12Resource> temporaryBuffer;
    ComPtr<ID3D12Resource> persistentBuffer;

    ID3D12DescriptorHeap** d3D12DescriptorHeaps;
    unsigned int d3D12DescriptorHeapsCount;
    unsigned int descriptorCount;
    UINT64 tensorBufferSize;

    UINT64 temporaryResourceSize;
    UINT64 persistentResourceSize;
};

static DmlAppGlobals _dml;


constexpr UINT tensorSizes[4] = { 1, 2, 3, 4 };
constexpr UINT tensorElementCount = tensorSizes[0] * tensorSizes[1] * tensorSizes[2] * tensorSizes[3];

void InitializeDirect3D12(
    ComPtr<ID3D12Device>& d3D12Device,
    ComPtr<ID3D12CommandQueue>& commandQueue,
    ComPtr<ID3D12CommandAllocator>& commandAllocator,
    ComPtr<ID3D12GraphicsCommandList>& commandList)
{
#if defined(_DEBUG) && !defined(_GAMING_XBOX)
    ComPtr<ID3D12Debug> d3D12Debug;
    // Throws if the D3D12 debug layer is missing - you must install the Graphics Tools optional feature
    THROW_IF_FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(d3D12Debug.GetAddressOf())));
    d3D12Debug->EnableDebugLayer();
#endif

    ComPtr<IDXGIFactory4> dxgiFactory;
    THROW_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(dxgiFactory.GetAddressOf())));

    ComPtr<IDXGIAdapter> dxgiAdapter;
    UINT adapterIndex{};
    HRESULT hr{};
    do
    {
        dxgiAdapter = nullptr;
        THROW_IF_FAILED(dxgiFactory->EnumAdapters(adapterIndex, dxgiAdapter.ReleaseAndGetAddressOf()));
        ++adapterIndex;

        hr = ::D3D12CreateDevice(
            dxgiAdapter.Get(),
            D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(d3D12Device.ReleaseAndGetAddressOf()));
        if (hr == DXGI_ERROR_UNSUPPORTED) continue;
        THROW_IF_FAILED(hr);
    } while (hr != S_OK);

    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    THROW_IF_FAILED(d3D12Device->CreateCommandQueue(
        &commandQueueDesc,
        IID_GRAPHICS_PPV_ARGS(commandQueue.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(d3D12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_GRAPHICS_PPV_ARGS(commandAllocator.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(d3D12Device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        commandAllocator.Get(),
        nullptr,
        IID_GRAPHICS_PPV_ARGS(commandList.ReleaseAndGetAddressOf())));
}



void CloseExecuteResetWait(
    ComPtr<ID3D12Device> d3D12Device,
    ComPtr<ID3D12CommandQueue> commandQueue,
    ComPtr<ID3D12CommandAllocator> commandAllocator,
    ComPtr<ID3D12GraphicsCommandList> commandList)
{
    THROW_IF_FAILED(commandList->Close());

    ID3D12CommandList* commandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

    ComPtr<ID3D12Fence> d3D12Fence;
    THROW_IF_FAILED(d3D12Device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_GRAPHICS_PPV_ARGS(d3D12Fence.GetAddressOf())));

    wil::unique_handle fenceEventHandle(::CreateEvent(nullptr, true, false, nullptr));
    THROW_LAST_ERROR_IF_NULL(fenceEventHandle);

    THROW_IF_FAILED(commandQueue->Signal(d3D12Fence.Get(), 1));
    THROW_IF_FAILED(d3D12Fence->SetEventOnCompletion(1, fenceEventHandle.get()));

    ::WaitForSingleObjectEx(fenceEventHandle.get(), INFINITE, FALSE);

    THROW_IF_FAILED(commandAllocator->Reset());
    THROW_IF_FAILED(commandList->Reset(commandAllocator.Get(), nullptr));
}

void InitializeDML()
{

}

void CompileOperators()
{
}


float test_multiply(float v)
{

    // Set up Direct3D 12.
    InitializeDirect3D12(_dml.d3D12Device, _dml.commandQueue, _dml.commandAllocator, _dml.commandList);

    // Create the DirectML device.

    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;

#if defined (_DEBUG)
    // If the project is in a debug build, then enable debugging via DirectML debug layers with this flag.
    dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif

    THROW_IF_FAILED(DMLCreateDevice(
        _dml.d3D12Device.Get(),
        dmlCreateDeviceFlags,
        IID_PPV_ARGS(_dml.dmlDevice.GetAddressOf())));

    // Create DirectML operator(s). Operators represent abstract functions such as "multiply", "reduce", "convolution", or even
    // compound operations such as recurrent neural nets. This example creates an instance of the Identity operator,
    // which applies the function f(x) = x for all elements in a tensor.

    //ComPtr<IDMLCompiledOperator> dmlCompiledOperator;

    dml::Graph graph(_dml.dmlDevice.Get());
    dml::TensorDesc::Dimensions dimensions(std::begin(tensorSizes), std::end(tensorSizes));
    dml::TensorDesc desc = { DML_TENSOR_DATA_TYPE_FLOAT32, dimensions };
    dml::Expression input = dml::InputTensor(graph, 0, desc);


    // The memory referenced by any constant nodes (e.g, "scalar" below) needs to be kept alive until the graph is compiled.
    float scalar = 3.4f;
    auto constValue = dml::ConstantData(
        graph,
        dml::Span<const dml::Byte>(reinterpret_cast<const dml::Byte*>(&scalar), sizeof(scalar)),
        dml::TensorDesc{ DML_TENSOR_DATA_TYPE_FLOAT32, {1} });

    // Creates the DirectMLX Graph then takes the compiled operator(s) and attaches it to the relative COM Interface.
    dml::Expression output = dml::Identity(input) * dml::Reinterpret(constValue, dimensions, dml::TensorStrides{ 0,0,0,0 });


    DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    _dml.dmlCompiledOperator.Attach(graph.Compile(executionFlags, { output }).Detach());

    // 24 elements * 4 == 96 bytes.
    _dml.tensorBufferSize = { desc.totalTensorSizeInBytes };



    IDMLCompiledOperator* dmlCompiledOperators[] = { _dml.dmlCompiledOperator.Get() };
    THROW_IF_FAILED(_dml.dmlDevice->CreateOperatorInitializer(
        ARRAYSIZE(dmlCompiledOperators),
        dmlCompiledOperators,
        IID_PPV_ARGS(_dml.dmlOperatorInitializer.GetAddressOf())));


    // Query the operator for the required size (in descriptors) of its binding table.
    // You need to initialize an operator exactly once before it can be executed, and
    // the two stages require different numbers of descriptors for binding. For simplicity,
    // we create a single descriptor heap that's large enough to satisfy them both.
    DML_BINDING_PROPERTIES initializeBindingProperties = _dml.dmlOperatorInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProperties = _dml.dmlCompiledOperator->GetBindingProperties();
    _dml.descriptorCount = std::max(
        initializeBindingProperties.RequiredDescriptorCount,
        executeBindingProperties.RequiredDescriptorCount);


    // Create descriptor heaps.


    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = _dml.descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(
        _dml.d3D12Device->CreateDescriptorHeap(
            &descriptorHeapDesc,
            IID_GRAPHICS_PPV_ARGS(_dml.descriptorHeap.GetAddressOf())));

    // Set the descriptor heap(s).
    ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = { _dml.descriptorHeap.Get() };
    _dml.commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    _dml.d3D12DescriptorHeaps = d3D12DescriptorHeaps;
    _dml.d3D12DescriptorHeapsCount = ARRAYSIZE(d3D12DescriptorHeaps);

    // Create a binding table over the descriptor heap we just created.
    DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    dmlBindingTableDesc.Dispatchable = _dml.dmlOperatorInitializer.Get();
    dmlBindingTableDesc.CPUDescriptorHandle = _dml.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.GPUDescriptorHandle = _dml.descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.SizeInDescriptors = _dml.descriptorCount;


    THROW_IF_FAILED(_dml.dmlDevice->CreateBindingTable(
        &dmlBindingTableDesc,
        IID_PPV_ARGS(_dml.dmlBindingTable.GetAddressOf())));


    // Create the temporary and persistent resources that are necessary for executing an operator.

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.

    _dml.temporaryResourceSize = std::max(
        initializeBindingProperties.TemporaryResourceSize,
        executeBindingProperties.TemporaryResourceSize);

    _dml.persistentResourceSize = executeBindingProperties.PersistentResourceSize;



    // Bind and initialize the operator on the GPU.

    if (_dml.temporaryResourceSize != 0)
    {
        THROW_IF_FAILED(
            _dml.d3D12Device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(_dml.temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(_dml.temporaryBuffer.GetAddressOf())));

        if (initializeBindingProperties.TemporaryResourceSize != 0)
        {
            DML_BUFFER_BINDING bufferBinding{ _dml.temporaryBuffer.Get(), 0, _dml.temporaryResourceSize };
            DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
            _dml.dmlBindingTable->BindTemporaryResource(&bindingDesc);
        }
    }

    if (_dml.persistentResourceSize != 0)
    {
        THROW_IF_FAILED(
            _dml.d3D12Device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(_dml.persistentResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(_dml.persistentBuffer.GetAddressOf())));

        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{ _dml.persistentBuffer.Get(), 0, _dml.persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        _dml.dmlBindingTable->BindOutputs(1, &bindingDesc);
    }

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    THROW_IF_FAILED(_dml.dmlDevice->CreateCommandRecorder(
        IID_PPV_ARGS(_dml.dmlCommandRecorder.GetAddressOf())));

    // Record execution of the operator initializer.
    _dml.dmlCommandRecorder->RecordDispatch(
        _dml.commandList.Get(),
        _dml.dmlOperatorInitializer.Get(),
        _dml.dmlBindingTable.Get());

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could
    // in principle record the execution into the same command list as the initialization, but you need only to Initialize
    // once, and typically you want to Execute an operator more frequently than that.
    CloseExecuteResetWait(
        _dml.d3D12Device,
        _dml.commandQueue,
        _dml.commandAllocator,
        _dml.commandList);

    // 
    // Bind and execute the operator on the GPU.
    // 

    //ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = { _dml.descriptorHeap.Get() };
    //_dml.commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    _dml.commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the
    // initializer).


    //SR make a new binding table
    /*DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    dmlBindingTableDesc.Dispatchable = _dml.dmlOperatorInitializer.Get();
    dmlBindingTableDesc.CPUDescriptorHandle = _dml.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.GPUDescriptorHandle = _dml.descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.SizeInDescriptors = _dml.descriptorCount;*/

    dmlBindingTableDesc.Dispatchable = _dml.dmlCompiledOperator.Get();

    THROW_IF_FAILED(_dml.dmlBindingTable->Reset(&dmlBindingTableDesc));

    if (_dml.temporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ _dml.temporaryBuffer.Get(), 0, _dml.temporaryResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        _dml.dmlBindingTable->BindTemporaryResource(&bindingDesc);
    }

    if (_dml.persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ _dml.persistentBuffer.Get(), 0, _dml.persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        _dml.dmlBindingTable->BindPersistentResource(&bindingDesc);
    }

    // Create tensor buffers for upload/input/output/readback of the tensor elements.

    ComPtr<ID3D12Resource> uploadBuffer;
    THROW_IF_FAILED(
        _dml.d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(_dml.tensorBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(uploadBuffer.GetAddressOf())));

    ComPtr<ID3D12Resource> inputBuffer;
    THROW_IF_FAILED(
        _dml.d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(_dml.tensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(inputBuffer.GetAddressOf())));

    std::wcout << std::fixed; std::wcout.precision(4);
    std::array<FLOAT, tensorElementCount> inputTensorElementArray;
    {
        std::wcout << L"input tensor: ";
        for (auto& element : inputTensorElementArray)
        {
            element = v;
            std::wcout << element << L' ';
        };
        std::wcout << std::endl;

        D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
        tensorSubresourceData.pData = inputTensorElementArray.data();
        tensorSubresourceData.RowPitch = static_cast<LONG_PTR>(_dml.tensorBufferSize);
        tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

        // Upload the input tensor to the GPU.
        ::UpdateSubresources(
            _dml.commandList.Get(),
            inputBuffer.Get(),
            uploadBuffer.Get(),
            0,
            0,
            1,
            &tensorSubresourceData);

        _dml.commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                inputBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            )
        );
    }

    DML_BUFFER_BINDING inputBufferBinding{ inputBuffer.Get(), 0, _dml.tensorBufferSize };
    DML_BINDING_DESC inputBindingDesc{ DML_BINDING_TYPE_BUFFER, &inputBufferBinding };
    _dml.dmlBindingTable->BindInputs(1, &inputBindingDesc);

    ComPtr<ID3D12Resource> outputBuffer;
    THROW_IF_FAILED(
        _dml.d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(_dml.tensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(outputBuffer.GetAddressOf())));

    DML_BUFFER_BINDING outputBufferBinding{ outputBuffer.Get(), 0, _dml.tensorBufferSize };
    DML_BINDING_DESC outputBindingDesc{ DML_BINDING_TYPE_BUFFER, &outputBufferBinding };
    _dml.dmlBindingTable->BindOutputs(1, &outputBindingDesc);

    // Record execution of the compiled operator.
    _dml.dmlCommandRecorder->RecordDispatch(_dml.commandList.Get(), _dml.dmlCompiledOperator.Get(), _dml.dmlBindingTable.Get());

    CloseExecuteResetWait(
        _dml.d3D12Device, _dml.commandQueue, _dml.commandAllocator, _dml.commandList);

    // The output buffer now contains the result of the identity operator,
    // so read it back if you want the CPU to access it.

    ComPtr<ID3D12Resource> readbackBuffer;
    THROW_IF_FAILED(
        _dml.d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(_dml.tensorBufferSize),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(readbackBuffer.GetAddressOf())));

    _dml.commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            outputBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE
        )
    );

    _dml.commandList->CopyResource(readbackBuffer.Get(), outputBuffer.Get());

    CloseExecuteResetWait(_dml.d3D12Device, _dml.commandQueue, _dml.commandAllocator, _dml.commandList);

    D3D12_RANGE tensorBufferRange{ 0, static_cast<SIZE_T>(_dml.tensorBufferSize) };
    FLOAT* outputBufferData{};
    THROW_IF_FAILED(readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBufferData)));

    std::wstring outputString = L"output tensor: ";
    for (size_t tensorElementIndex{ 0 }; tensorElementIndex < tensorElementCount; ++tensorElementIndex, ++outputBufferData)
    {
        outputString += std::to_wstring(*outputBufferData) + L' ';
    }

    std::wcout << outputString << std::endl;
    OutputDebugStringW(outputString.c_str());

    D3D12_RANGE emptyRange{ 0, 0 };
    readbackBuffer->Unmap(0, &emptyRange);

    return 0.f;
}


void _init_dml()
{
    // Initialize DirectML.
    InitializeDML();

    CompileOperators();

}

void _test_dml(const float v)
{
    // Run the DirectML test.
    float result1 = test_multiply(v);
}

void _close_dml()
{
    // Clean up DirectML.
    _dml.dmlDevice.Reset();
    _dml.commandList.Reset();
    _dml.commandAllocator.Reset();
    _dml.commandQueue.Reset();
    _dml.d3D12Device.Reset();
}

