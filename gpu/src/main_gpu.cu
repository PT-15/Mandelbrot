#include <iostream>

#include <math.h>
#include <unistd.h>

#include <SDL2/SDL.h>
#include "/home/morgana/programs/libraries/myCUDAextensions.h"

#define WIDTH 1280
#define HEIGHT 720

//Paint it pink: 255 20 147
#define R 255
#define G 20
#define B 147

struct mandelConfig {
    float offsetX; //width/2
    float offsetY; //height/2
    float scale;
    int maxIt;
};

struct complexNum {
    float r;
    float i;
    __device__ complexNum(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2 (void) {
        return r*r + i*i;
    }
    __device__ complexNum operator* (const complexNum& a){
        return complexNum(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ complexNum operator+(const complexNum& a) {
        return complexNum(r+a.r, i+a.i);
    }
};

__device__ int mandelbrot (int x, int y, const mandelConfig& pConf)
{
    const float maxZ = 1 + sqrt(2.);

    complexNum z(0,0);

    float fx = (float)x * pConf.scale + pConf.offsetX - (pConf.scale * (float)WIDTH/2.);
    float fy = (float)y * pConf.scale + pConf.offsetY - (pConf.scale * (float)HEIGHT/2.);

    //float fx = ((x - (float)WIDTH/2) - pConf.offsetX)/pConf.scale;
    //float fy = ((y - (float)HEIGHT/2) - pConf.offsetY)/pConf.scale;

    complexNum c(fx, fy);

    for (int i = 0; i < pConf.maxIt; i++){
        z = z*z + c;
        if (sqrt(z.magnitude2()) > maxZ)
            return i;
    }
    return pConf.maxIt;
}

__global__ void kernel (const mandelConfig pConf, int* grid, uint8_t rShift, uint8_t gShift, uint8_t bShift)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    int i = mandelbrot(x, y, pConf);

    int r = (R*i)/pConf.maxIt;
    int g = (G*i)/pConf.maxIt;
    int b = (B*i)/pConf.maxIt;
    int offset = x + y*gridDim.x;

    grid[offset] = (r << rShift) + (g << gShift) + (b << bShift);
}

int main ( void )
{
    //Initialize SDL
    uint32_t flags = SDL_INIT_VIDEO;
    SDL_version ver;

    SDL_Init(flags);
    SDL_GetVersion(&ver);
    std::cout << "SDL version " << (int)ver.major << "." << (int)ver.minor << "." << (int)ver.patch << "\n";
    
    //Create SDL surface
    int x = 0;
    int y = 0;
    int width = (int)WIDTH;
    int height = (int)HEIGHT;

    SDL_Window* win = SDL_CreateWindow("ULTIMATE MANDELBROT MASSIVE PARALLEL PROCESSING!!!", x, y, width, height, 0);
    SDL_Surface* surf = SDL_GetWindowSurface(win);
    int* dev_grid;

    //Create first call configuration
    mandelConfig pConf;
    pConf.offsetX = 0.;
    pConf.offsetY = 0.;
    pConf.scale = 0.02;
    pConf.maxIt = 50;

    //Allocate space in device for the pixel grid
    HANDLE_ERROR( cudaMalloc((void**)&dev_grid, width*height*sizeof(int)) );

    //Execute first kernel
    dim3 grid(width, height);
    kernel<<<grid,1>>>(pConf, dev_grid, surf->format->Rshift, surf->format->Gshift, surf->format->Bshift);

    //Copy pixel grid to SDL surface
    SDL_LockSurface(surf);
    HANDLE_ERROR( cudaMemcpy(surf->pixels, dev_grid, width*height*sizeof(int), cudaMemcpyDeviceToHost) );
    SDL_UnlockSurface(surf);

    //Update screen
    SDL_UpdateWindowSurface(win);
    
    //Start input check loop
    bool done = false;
    SDL_Event ev;
    while (!done && SDL_WaitEvent(&ev)){
        switch(ev.type){
            case SDL_QUIT:
                done = true;
                break;
            case SDL_KEYDOWN:
                {
                    SDL_KeyboardEvent* key = (SDL_KeyboardEvent*)&ev;
                    switch(key->keysym.sym){
                        case SDLK_ESCAPE:
                            done = true;
                            break;
#define OFFSET_STEP 100.
                        case SDLK_RIGHT:
                            pConf.offsetX += OFFSET_STEP*pConf.scale;
                            break;
                        case SDLK_LEFT:
                            pConf.offsetX -= OFFSET_STEP*pConf.scale;
                            break;
                        case SDLK_UP:
                            pConf.offsetY -= OFFSET_STEP*pConf.scale;
                            break;
                        case SDLK_DOWN:
                            pConf.offsetY += OFFSET_STEP*pConf.scale;
                            break;
                        case SDLK_PLUS:
                            pConf.scale /= 1.1;
                            break;
                        case SDLK_MINUS:
                            if (pConf.scale + 10. > 0.)
                                pConf.scale *= 1.1;
                            break;
                        case SDLK_l:
                            if (pConf.maxIt - 10 > 0.)
                                pConf.maxIt -= 10;
                            break;
                        case SDLK_m:
                            if (pConf.maxIt <= 300)
                                pConf.maxIt += 10;
                            break;
                    }

                    //Calculate new pixel values
                    kernel<<<grid,1>>>(pConf, dev_grid, surf->format->Rshift, surf->format->Gshift, surf->format->Bshift);

                    //Copy pixel grid to SDL surface
                    SDL_LockSurface(surf);
                    HANDLE_ERROR( cudaMemcpy(surf->pixels, dev_grid, width*height*sizeof(int), cudaMemcpyDeviceToHost) );
                    SDL_UnlockSurface(surf);

                    //Update screen
                    SDL_UpdateWindowSurface(win);
                    
                    std::cout << "X offset: " << pConf.offsetX << "\nY offset: " << pConf.offsetY << "\nScale: " << pConf.scale << "\nMax iterations: " << pConf.maxIt << "\n";
                }
        }
    }

    //Free memory
    cudaFree(dev_grid);

}
