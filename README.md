# README
Instalacja sdl na miracle:
w tym samym katalogu w którym jest n-body-simulation

hg clone https://hg.libsdl.org/SDL SDL
cd SDL
mkdir build
cd build
../configure --prefix=(bezwzględna scieżka do katalogu w którym jet folder n-body-simulation)
make
sudo make install

w makefile trzeba podmienić ścieżkę do bibliotek uruchamiając skrypt sdl2-config --cflags --libs

Założenia projektu:

1. Wersja w n^2 na cuda
2. Wersja nlogn (Barnes–Hut simulation)
3. wersja na procesorze
4. porównanie czasu działania jakiś wykresik.