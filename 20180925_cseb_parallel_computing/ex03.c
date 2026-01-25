/*
拡散方程式の初期値問題に関するMPIを用いたプログラム
今回は1次元配列
2018年9月20日
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define NX 193
#define NY 193
#define W (NX-1)
#define ND ((NX+1)*(NY+1))
#define NXNY (NX*NY)
//拡散係数
#define D 1
//ステップ回数
#define N 40000
//全MPI(プロセス集合)を表す変数
#define MCW MPI_COMM_WORLD

int main(int argc, char **argv){
	//np: プロセス集合全体の大きさ
	//me: プロセス集合の中の自分のid
	int np,me;
	MPI_Status st;

	//初期化
	MPI_Init(&argc,&argv);
	//np,meの計算
	MPI_Comm_size(MCW,&np);
  	MPI_Comm_rank(MCW,&me);

  	//一次元分割したときの縦の長さ
  	int ny=(NY-1)/np;
  	//printf("np=%d,ny=%d\n",np,ny);

  	//部分領域交換のための変数
  	//ココ微妙に変わるかも
	int north = me<np-1 ? me+1 : MPI_PROC_NULL;
    int south = me>0    ? me-1 : MPI_PROC_NULL;

    //ループの変数
    int i,j,t,r;

    /*
	ここらへんをはじめにMPIに載る形で並列化しなければいけない。
	//現在の分布(最初に初期化している)
	double u[NY+1][NX+1] = {0.0};
	//次の時間ステップでの分布
	double un[NY+1][NX+1];
	*/


	//刻み幅
	double dx = 1.0/NX;
	//時間刻み幅は 0.1*(空間の刻み幅)^2
	double dt = D*dx*dx*0.1;
	double mu = D*dt/dx/dx;


    //配列を定義
    double (*u)[NX+1];
    u=(double(*)[NX+1])malloc((ny+2)*(NX+1)*sizeof(double));
    double (*un)[NX+1];
    un=(double(*)[NX+1])malloc((ny+2)*(NX+1)*sizeof(double));

	/*
	境界条件の設定
	//境界値の設定
	//un(y,0,t)=0.5 for 0<y<=1
	for(i=1;i<NY+1;i++){
		un[i][0] = 0.5;
	}
	//un(y,1,t)=0.0 for 0<y<1
	for(i=1;i<NY;i++){
		un[i][NX] = 0.0;
	}
	//un(0,x,t)=1.0 for 0<=x<=1
	for(i=0;i<NX+1;i++){
		un[0][i] = 1.0;
	}
	//un(1,x,t)=0.0 for 0<x<=1
	for(i=1;i<NX+1;i++){
		un[NY][i] = 0.0;
	}
	*/

    //uの初期化
    for(j=0;j<ny+2;j++){	
		for(i=0;i<NX+1;i++){
			u[j][i]=0.0;
		}
	}

    //x方向の境界条件の設定
    for(i=0;i<ny+2;i++){
    	un[i][0]=0.5;
    	un[i][NX]=0.0;
    }
    //y方向の境界条件の設定
    if(me==0){
    	for(i=0;i<NX+1;i++){
    		un[0][i]=1.0;
    	}
    }
    if(me==np-1){
    	for(i=1;i<NX+1;i++){
    		un[ny+1][i]=0.0;
    	}
    }

	//値の逐次更新
	for(t=0;t<N;t++){
		//ここで値の交換!!
		//まずは北側の交換
		MPI_Sendrecv(u[ny],NX+1,MPI_DOUBLE,north,0,u[ny+1],NX+1,MPI_DOUBLE,north,0,MCW,&st);
		//次に南側の交換
		MPI_Sendrecv(u[1],NX+1,MPI_DOUBLE,south,0,u[0],NX+1,MPI_DOUBLE,south,0,MCW,&st);

		//更新
		for(j=1;j<=ny;j++){
			for(i=1;i<NX;i++){
				un[j][i]=u[j][i]+mu*(u[j][i+1]+u[j][i-1]+u[j+1][i]+u[j-1][i]-4.0*u[j][i]);
			}
		}
		//unの値をuにコピーする！
		for(j=0;j<ny+2;j++){	
			for(i=0;i<NX+1;i++){
				u[j][i]=un[j][i];
			}
		}		
	}

	//ファイル出力
	//とりあえずの下手くそな出力方法らしい。。
	/*
	FILE *udata;
	udata = fopen("u.data","w");
	for (j=0; j<=NY; j++){
		for (i=0; i<=NX; i++)
			fprintf(udata, " %.15E %.15E %.15E\n", i*dx, j*dx, u[j][i]);
		fprintf(udata, "\n");
	}
	*/
	FILE *udata;
	if(me==0){
		udata = fopen("ex03.data","w");
		//まずは自分の領域を出力
		for(j=0;j<ny+1;j++){
			for(i=0;i<NX+1;i++)
				fprintf(udata," %.15E %.15E %.15E\n", i*dx, j*dx, u[j][i]);
			fprintf(udata,"\n");
		}
		//次に他のidのひとから順に受信して出力する
		for(r=1;r<np-1;r++){
			//r番の人から受信する
			MPI_Recv(&u[1][0],(ny+1)*(NX+1),MPI_DOUBLE,r,0,MCW,&st);
			//出力する
			for(j=1;j<ny+1;j++){
				for(i=0;i<NX+1;i++)
					fprintf(udata," %.15E %.15E %.15E\n", i*dx, (j+r*ny)*dx, u[j][i]);
				fprintf(udata,"\n");	
			}
		}
		//北側の境界を考慮して最後だけ別に出力させる
		//最後のヤツの受信
		MPI_Recv(&u[1][0],(ny+1)*(NX+1),MPI_DOUBLE,np-1,0,MCW,&st);
		//出力
		for(j=1;j<ny+2;j++){
			for(i=0;i<NX+1;i++)
				fprintf(udata," %.15E %.15E %.15E\n", i*dx, (j+r*ny)*dx, u[j][i]);
			fprintf(udata,"\n");
		}
		//closeして終了
		fclose(udata);
	} else
		//0番目のidのひとに投げる
		MPI_Send(&u[1][0],(ny+1)*(NX+1),MPI_DOUBLE,0,0,MCW);


	MPI_Finalize();

}