//
// Created by peter on 2/1/17.
//

#ifndef N_BODY_SIMULATION_GRAPHIC_H
#define N_BODY_SIMULATION_GRAPHIC_H
#include "../include/SDL2/SDL.h"
extern const int SCREEN_WIDTH;
extern const int SCREEN_HEIGHT;

extern SDL_Window* gWindow;
extern SDL_Surface* gHelloWorld;
extern SDL_Renderer* gRenderer;

bool init();
bool loadMedia();
void close();
void renderSquare();
void draw_circle(SDL_Point center, int radius, SDL_Color color);
void clearScreen();
void updateRender();

const SDL_Color red = {255, 1, 1, 255};
const SDL_Color black = {0, 0, 0, 255};
const SDL_Color yellow = {255, 255, 0, 255};

#endif  // N_BODY_SIMULATION_GRAPHIC_H
