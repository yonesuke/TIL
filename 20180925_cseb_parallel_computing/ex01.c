/*
拡散方程式の初期値問題に関する逐次プログラム
2018年9月18日
*/

#include <stdio.h>
#include <time.h>

#define NX 193
#define NY 193
#define ND ((NX+1)*(NY+1))
#define NXNY (NX*NY)
//拡散係数
#define D 1
//ステップ回数
#define N 40000

int main(void){
	//現在の分布(最初に初期化している)
	double u[NY+1][NX+1];
	//次の時間ステップでの分布
	double un[NY+1][NX+1];
	//刻み幅
	double dx = 1.0/NX;
	//時間刻み幅は 0.1*(空間の刻み幅)^2
	double dt = D*dx*dx*0.1;
	double mu = D*dt/dx/dx;

	//ファイル出力
	FILE *udata;
	udata = fopen("ex01.data","w");

	//時間計測
	clock_t start,end;

	int i,j,t;

	//uの初期化
    for(j=0;j<NY+1;j++){	
		for(i=0;i<NX+1;i++){
			u[j][i]=0.0;
		}
	}

	//境界値の設定
	//un(0,y,t)=0.5 for 0<y<=1
	for(i=1;i<NY+1;i++){
		un[i][0] = 0.5;
	}
	//un(1,y,t)=0.0 for 0<y<1
	for(i=1;i<NY;i++){
		un[i][NX] = 0.0;
	}
	//un(x,0,t)=1.0 for 0<=x<=1
	for(i=0;i<NX+1;i++){
		un[0][i] = 1.0;
	}
	//un(x,1,t)=0.0 for 0<x<=1
	for(i=1;i<NX+1;i++){
		un[NY][i] = 0.0;
	}
	
	start = clock();

	//値の逐次更新
	for(t=0;t<N;t++){
		//更新
		for(j=1;j<NY;j++){	
			for(i=1;i<NX;i++){
				un[j][i]=u[j][i]+mu*(u[j][i+1]+u[j][i-1]+u[j+1][i]+u[j-1][i]-4.0*u[j][i]);
			}
		}
		//unの値をuにコピーする！
		for(j=0;j<NY+1;j++){	
			for(i=0;i<NX+1;i++){
				u[j][i]=un[j][i];
			}
		}		
	}

	end = clock();

	//値の出力
	for (j=0; j<=NY; j++){
		for (i=0; i<=NX; i++)
			fprintf(udata, " %.15E %.15E %.15E\n", i*dx, j*dx, u[j][i]);
		fprintf(udata, "\n");
	}

	printf("%.2fseconds\n",(double)(end-start)/CLOCKS_PER_SEC);

}