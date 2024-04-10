#pragma once

#include "imgui.h"
#include "imgui_impl_win32.h"
#include <d3d11.h>

class App
{
public:
    HWND hwnd;
    WNDCLASSEXW wc;
    const ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    virtual ~App() = default;

    virtual bool OnCreate() { return true; }
    virtual bool OnDraw() { return true; }
    virtual void OnDestroy() {}

    int Init(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow);

    void Run();

    void Cleanup();

    ID3D11Device* GetDevice();
};

ID3D11ShaderResourceView* CreateTexture2DFromImageFile(ID3D11Device* device, const char* filename);