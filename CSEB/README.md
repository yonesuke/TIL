# 計算科学演習B

スパコンを使って拡散方程式の初期値境界値問題を解かせる授業。
具体的には逐次計算、OpenMP、MPIを用いて高速計算を行わせた。

## [ex01.c](./ex01.c)
はじめに逐次計算で拡散方程式を数値計算した。
京大のスパコンで実行する場合は、
```
icc -O0 ex01.c
./a.out
```
で実行できる。結果は`ex01.data`に出力される。

## [ex03.c](./ex03.c)
MPIを用いて一次元分割で拡散方程式を解かせた。
京大のスパコンで実行する場合は、
```
mpiicc -O0 -xHost -o hoge ex03.c
tssrun -q grxxxxxx -ug axxxxxxx -A p=4 mpiexec.hydra ./hoge
```
で実行できる。結果は`ex03.data`に出力される。

## [ex04.c](./ex04.c)
MPIを用いて二次元分割で拡散方程式を解かせた。
京大のスパコンで実行する場合は、
```
mpiicc -O0 -xHost -o hoge ex04.c
tssrun -q grxxxxxx -ug axxxxxxx -A p=16 mpiexec.hydra ./hoge
```
で実行できる。結果は`ex04.data`に出力される。
