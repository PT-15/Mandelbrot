#include <iostream>

#include <math.h>
#include <unistd.h>

#include <SDL2/SDL.h>
#include "/home/morgana/programs/libraries/myCUDAextensions.h"

#define WIDTH 1280
#define HEIGHT 720

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
    complexNum c( (x - pConf.offsetX)/pConf.scale, (y - pConf.offsetY)/pConf.scale );

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

    int r = (255*i)/pConf.maxIt;
    int g = (20*i)/pConf.maxIt;
    int b = (147*i)/pConf.maxIt;
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
    int width = WIDTH;
    int height = HEIGHT;

    SDL_Window* win = SDL_CreateWindow("ULTIMATE MANDELBROT MASSIVE PARALLEL PROCESSING!!!", x, y, width, height, 0);
    SDL_Surface* surf = SDL_GetWindowSurface(win);
    int* dev_grid;

    //Create first call configuration
    mandelConfig pConf;
    pConf.offsetX = width/2;
    pConf.offsetY = height/2;
    pConf.scale = 10;
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
                        case SDLK_RIGHT:
                            pConf.offsetX += 20.;
                            break;
                        case SDLK_LEFT:
                            pConf.offsetX -= 20.;
                            break;
                        case SDLK_UP:
                            pConf.offsetY -= 20.;
                            break;
                        case SDLK_DOWN:
                            pConf.offsetY += 20.;
                            break;
                        case SDLK_PLUS:
                            pConf.scale *= 1.1;
                            break;
                        case SDLK_MINUS:
                            if (pConf.scale -10. > 0.)
                                pConf.scale /= 1.1;
                            break;
                        case SDLK_l:
                            if (pConf.maxIt - 10 > 0.)
                                pConf.maxIt -= 10;
                            break;
                        case SDLK_m:
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
