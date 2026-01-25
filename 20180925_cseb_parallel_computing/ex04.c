/*
拡散方程式の初期値問題に関するMPIを用いたプログラム
今回は2次元配列!!
2018年9月20日
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define NX 193
#define NY 193
#define W (NX-1)
//フィアル出力する際の一行あたりの文字数
#define LW 67
#define ND ((NX+1)*(NY+1))
#define NXNY (NX*NY)
//拡散係数
#define D 1
//ステップ回数
#define N 40000
//全MPI(プロセス集合)を表す変数
#define MCW MPI_COMM_WORLD

int main(int argc, char **argv){
    
    //ループの変数
    int i,j,k,t;

    //刻み幅
	double dx = 1.0/NX;
	//時間刻み幅は 0.1*(空間の刻み幅)^2
	double dt = D*dx*dx*0.1;
	double mu = D*dt/dx/dx;

	//np: プロセス集合全体の大きさ
	//me: プロセス集合の中の自分のid
	int np,me;
	MPI_Status st;

	//初期化
	MPI_Init(&argc,&argv);
	//np,meの計算
	MPI_Comm_size(MCW,&np);
  	MPI_Comm_rank(MCW,&me);

  	//2次元分割!!
  	//プロセス数(np)を平方数にしておくと
  	//相加平均と相乗平均の関係から
  	//縦横の分割はおなじになってくれる!!
  	//今回は万が一、npが平方数じゃないときのために
  	//MPI_Dims_createを使うことにする。
  	int dims[2]={0,0};
  	MPI_Dims_create(np,2,dims);

  	//分割した1ブロックごとの大きさ
  	int nx=(NX-1)/dims[0];
  	int ny=(NY-1)/dims[1];

  	//直交プロセス空間
  	//プロセス座標系を生成する
  	//今回は非周期境界条件を設けているので
  	//periods配列は2次元とも0
  	int periods[2]={0,0};
  	MPI_Comm cart;
  	MPI_Cart_create(MCW,2,dims,periods,0,&cart);
  	//自分の座標は？？
  	int c[2];
  	MPI_Cart_coords(cart,me,2,c);
  	//隣接プロセスの座標の計算
  	int north,south,east,west;
  	MPI_Cart_shift(cart,0,1,&south,&north);
  	MPI_Cart_shift(cart,1,1,&west,&east);

    //配列を定義
    double u[ny+2][nx+2],un[ny+2][nx+2];

    //uの初期化
    for(j=0;j<ny+2;j++){	
		for(i=0;i<nx+2;i++){
			u[j][i]=0.0;
		}
	}
	//unの初期化
	//x方向の境界条件の設定
	//左端
	if(c[1]==0){
		for(i=0;i<ny+2;i++){
			un[i][0]=0.5;
		}
	}
	//右端
	if(c[1]==dims[1]-1){
		for(i=0;i<ny+2;i++){
			un[i][nx+1]=0.0;
		}
	}
	//y方向の境界条件の設定
	//下端
	if(c[0]==0){
		for(i=0;i<nx+2;i++){
			un[0][i]=1.0;
		}
	}
	//上端
	if(c[0]==dims[0]-1){
		if(c[1]==0){
			for(i=1;i<nx+2;i++){
				un[ny+1][i]=0.0;
			}
		}else{
			for(i=0;i<nx+2;i++){
				un[ny+1][i]=0.0;
			}
		}
	}

	//東西の値の交換のための派生データ型の生成
	MPI_Datatype vedge;
	MPI_Type_vector(ny,1,nx+2,MPI_DOUBLE,&vedge);
	MPI_Type_commit(&vedge);

	//値の逐次更新
	for(t=0;t<N;t++){
		//ここで値の交換!!
		//まずは北側の交換
		MPI_Sendrecv(&u[ny][1],nx,MPI_DOUBLE,north,0,&u[ny+1][1],nx,MPI_DOUBLE,north,0,MCW,&st);
		//次に南側の交換
		MPI_Sendrecv(&u[1][1],nx,MPI_DOUBLE,south,0,&u[0][1],nx,MPI_DOUBLE,south,0,MCW,&st);
		//東側の交換
		MPI_Sendrecv(&u[1][nx],1,vedge,east,0,&u[1][nx+1],1,vedge,east,0,MCW,&st);
		//最後に西側の交換
		MPI_Sendrecv(&u[1][1],1,vedge,west,0,&u[1][0],1,vedge,west,0,MCW,&st);

		//更新
		for(j=1;j<=ny;j++){
			for(i=1;i<=nx;i++){
				un[j][i]=u[j][i]+mu*(u[j][i+1]+u[j][i-1]+u[j+1][i]+u[j-1][i]-4.0*u[j][i]);
			}
		}
		//unの値をuにコピーする！
		for(j=0;j<ny+2;j++){	
			for(i=0;i<nx+2;i++){
				u[j][i]=un[j][i];
			}
		}		
	}

	//ファイル出力
	//最初の準備
	MPI_File udata;
	MPI_File_open(cart,"ex04.data",MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&udata);
	MPI_File_set_size(udata,(MPI_Offset)0);
	//filetypeの生成
	MPI_Datatype ftype;
	int size[2]={NY+1,LW*(NX+1)+1}, subsize[2], start[2], flag[2]={1,1};
	subsize[0]=ny;	subsize[1]=LW*nx;
	start[0]=c[0]*ny+1;	start[1]=LW*(c[1]*nx+1);
	if(c[0]==0){subsize[0]++;	start[0]=0;	flag[0]=0;}
	if(c[0]==dims[0]-1)	subsize[0]++;
	if(c[1]==0){subsize[1]+=LW;	start[1]=0;	flag[1]=0;}
	if(c[1]==dims[1]-1) subsize[1]+=LW+1;
	MPI_Type_create_subarray(2,size,subsize,start,MPI_ORDER_C,MPI_CHAR,&ftype);
	MPI_Type_commit(&ftype);
	MPI_File_set_view(udata,(MPI_Offset)0,MPI_CHAR,ftype,"native",MPI_INFO_NULL);
	//書き込み
	char *wbuf;
	wbuf=(char*)malloc((LW*(nx+2)+2)*sizeof(char));
	int max_x=(c[1]==dims[1]-1)?nx+1:nx;
	int max_y=flag[0]+subsize[0];
	for(j=flag[0];j<max_y;j++){
		for(i=flag[1],k=0;i<=max_x;i++,k+=LW){
			sprintf(wbuf+k," %.15E %.15E %.15E\n",(i+c[1]*nx)*dx,(j+c[0]*ny)*dx,u[j][i]);
		}
		if(c[1]==dims[1]-1)
			sprintf(wbuf+(k++),"\n");
		MPI_File_write(udata,wbuf,k,MPI_CHAR,&st);
	}

	//closeして終了
	MPI_File_close(&udata);

	MPI_Finalize();

}