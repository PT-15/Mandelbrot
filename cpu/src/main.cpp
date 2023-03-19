#include <iostream>

#include <math.h>
#include <unistd.h>

#include <SDL2/SDL.h>

struct complexNum {
    float real, imaginary;
};

complexNum complexSum (const complexNum& n1, const complexNum& n2){
    return {n1.real + n2.real, n1.imaginary + n2.imaginary};
}
complexNum complexProduct (const complexNum& n1, const complexNum& n2){
    complexNum tmp = {0, 0};
    tmp.real += n1.real * n2.real;
    tmp.imaginary += n1.imaginary * n2.real + n1.real * n2.imaginary;
    tmp.real += n1.imaginary * n2.imaginary * -1;
    return tmp;
}

void changePixelColour(SDL_Surface* surf, int* pixel, uint8_t r, uint8_t g, uint8_t b)
{
    //*pixel = (r << surf->format->Rshift) + (b << surf->format->Bshift) + (g << surf->format->Gshift);
    *pixel = SDL_MapRGB(surf->format, r, g, b);
}

void drawCircle(SDL_Surface* surf, int* pixel, float fx, float fy)
{
    int r = 0;
    int g = 0;
    int b = 255;

    float radius = 2.;
    if (sqrt(fx*fx + fy*fy) > radius){
        changePixelColour(surf, pixel, 0, 0, 0);
        return;
    }
    changePixelColour(surf, pixel, r, g, b);
}

void drawMandelbrot(SDL_Surface* surf, int* pixel, float fx, float fy, int maxIt)
{
    complexNum z = {0, 0};
    complexNum c = {fx, fy};
    float maxZ = 1. + sqrt(2.);
    int i;
    for (i = 0; i < maxIt; i++){
        z = complexSum(complexProduct(z, z), c);
        if (sqrt(z.real*z.real + z.imaginary*z.imaginary) > maxZ){
            break;
        }
    }
    int colour = (255*i)/maxIt;
    changePixelColour(surf, pixel, colour, colour, colour);
}

void draw_some_stuff (SDL_Surface* surf, float offsetX, float offsetY, float scale, int maxIt)
{
    SDL_LockSurface(surf);

    int* currPixel = (int*)surf->pixels;
    
    for (int y = 0; y < surf->h; y++){
        for (int x = 0; x < surf->w; x++){
            float fx = (x - offsetX)/scale;
            float fy = (y - offsetY)/scale;
            drawMandelbrot(surf, currPixel, fx, fy, maxIt);
//            drawCircle(surf, currPixel, fx, fy);
            currPixel++;
        }
    }

    SDL_UnlockSurface(surf);
}

int main()
{
    uint32_t flags = SDL_INIT_VIDEO;
    SDL_version ver;

    SDL_Init(flags);
    SDL_GetVersion(&ver);
    std::cout << "SDL version " << (int)ver.major << "." << (int)ver.minor << "." << (int)ver.patch << "\n";

    int x = 0;
    int y = 0;
    int width = 1280;
    int height = 720;
    float scale = 1.;
    float offsetX = 100;
    float offsetY = 100;
    int maxIt = 50;
    
    SDL_Window* win = SDL_CreateWindow("Mandelbrot magickness", x, y, width, height, 0);
    SDL_Surface* screenSurface = SDL_GetWindowSurface(win);
    draw_some_stuff(screenSurface, offsetX, offsetY, scale, maxIt);
    SDL_UpdateWindowSurface(win);

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
                            offsetX += 20.;
                            break;
                        case SDLK_LEFT:
                            offsetX -= 20.;
                            break;
                        case SDLK_UP:
                            offsetY -= 20.;
                            break;
                        case SDLK_DOWN:
                            offsetY += 20.;
                            break;
                        case SDLK_PLUS:
                            scale += 10.;
                            break;
                        case SDLK_MINUS:
                            if (scale -10. > 0.)
                                scale -= 10.;
                            break;
                        case SDLK_l:
                            if (maxIt -10. > 0.)
                                maxIt -= 10.;
                            break;
                        case SDLK_m:
                            maxIt += 10;
                            break;
                    }
                    draw_some_stuff(screenSurface, offsetX, offsetY, scale, maxIt);
                    SDL_UpdateWindowSurface(win);
                std::cout << "X offset: " << offsetX << "\nY offset: " << offsetY << "\nScale: " << scale << "\nMax iterations: " << maxIt << "\n";
                }
        }
    }
    return 0;
}