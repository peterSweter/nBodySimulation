//
// Created by peter on 2/1/17.
//
#include "graphic.h"

const int SCREEN_WIDTH = 1300;
const int SCREEN_HEIGHT = 1000;

// The window we'll be rendering to
SDL_Window *gWindow;

// The image we will load and show on the screen
SDL_Surface *gHelloWorld;

// The window renderer
SDL_Renderer *gRenderer = NULL;

bool init() {
  // Initialization flag
  bool success = true;

  // Initialize SDL
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    success = false;
  } else {
    // Create window
    gWindow = SDL_CreateWindow("N-body simulation", SDL_WINDOWPOS_UNDEFINED,
                               SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH,
                               SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (gWindow == NULL) {
      printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
      success = false;
    } else {
      gRenderer = SDL_CreateRenderer(gWindow, -1, NULL);
    }
  }

  return success;
}

bool loadMedia() {
  // Loading success flag
  bool success = true;

  return success;
}

void close() {
  // Deallocate surface
  SDL_FreeSurface(gHelloWorld);
  gHelloWorld = NULL;

  // Destroy window
  SDL_DestroyRenderer(gRenderer);
  gRenderer = NULL;
  SDL_DestroyWindow(gWindow);
  gWindow = NULL;
  // Quit SDL subsystems
  SDL_Quit();
}

void draw_circle(SDL_Point center, int radius, SDL_Color color) {

  SDL_SetRenderDrawColor(gRenderer, color.r, color.g, color.b, color.a);
  for (int w = 0; w < radius * 2; w++) {
    for (int h = 0; h < radius * 2; h++) {
      int dx = radius - w; // horizontal offset
      int dy = radius - h; // vertical offset
      if ((dx * dx + dy * dy) <= (radius * radius)) {
        SDL_RenderDrawPoint(gRenderer, center.x + dx, center.y + dy);
      }
    }
  }
}

void clearScreen() {
  // Clear screen
  SDL_SetRenderDrawColor(gRenderer, 0x00, 0x00, 0x00, 0xFF);
  SDL_RenderClear(gRenderer);
}

void renderSquare() {
  clearScreen();

  // Render red filled quad
  SDL_Rect fillRect = {SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2,
                       SCREEN_HEIGHT / 2};
  SDL_SetRenderDrawColor(gRenderer, 0xFF, 0x00, 0x00, 0xFF);
  SDL_RenderFillRect(gRenderer, &fillRect);

  // Render green outlined quad
  SDL_Rect outlineRect = {SCREEN_WIDTH / 6, SCREEN_HEIGHT / 6,
                          SCREEN_WIDTH * 2 / 3, SCREEN_HEIGHT * 2 / 3};
  SDL_SetRenderDrawColor(gRenderer, 0x00, 0xFF, 0x00, 0xFF);
  SDL_RenderDrawRect(gRenderer, &outlineRect);

  // Draw blue horizontal line
  SDL_SetRenderDrawColor(gRenderer, 0x00, 0x00, 0xFF, 0xFF);
  SDL_RenderDrawLine(gRenderer, 0, SCREEN_HEIGHT / 2, SCREEN_WIDTH,
                     SCREEN_HEIGHT / 2);

  // Draw vertical line of yellow dots
  SDL_SetRenderDrawColor(gRenderer, 0xFF, 0xFF, 0x00, 0xFF);
  for (int i = 0; i < SCREEN_HEIGHT; i += 4) {
    SDL_RenderDrawPoint(gRenderer, SCREEN_WIDTH / 2, i);
  }

  draw_circle({100, 10}, 50, {255, 1, 1, 255});

  // Update screen
  SDL_RenderPresent(gRenderer);

  return;
}

void updateRender() {
  // Update screen
  SDL_RenderPresent(gRenderer);

  return;
}
