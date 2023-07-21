#include <math.h>                   // 数学函数库，包含数学计算相关的函数
#include <omp.h>                    // OpenMP 头文件，支持并行计算
#include <stdio.h>                  // 标准输入输出函数库
#include <string.h>                 // 字符串处理函数库
#include <cblas.h>                  // CBLAS 头文件，用于调用 BLAS（Basic Linear Algebra Subprograms）库中的函数
#include <vector>                   // 向量容器库，用于定义和操作向量
#include <chrono>                   // 时间库，用于计算程序运行时间
#include <fstream>                  // 文件流库，用于文件读写操作
#include <iostream>                 // 输入输出流库
#include <immintrin.h>              // Intel SIMD（Single Instruction, Multiple Data）

指令集头文件，用于向量化指令

#include <thread>
#include <cstdlib>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;     // 时间点类型

auto max_threads=std::thread::hardware_concurrency();

int v_num = 0;                      // 顶点数量
int e_num = 0;                      // 边数量
int F0 = 0, F1 = 0, F2 = 0;         // 特征维度

vector<int> row_ptr;                // 行指针数组，CSR 格式中每一行起始位置的索引
vector<int> col_idx;                // 列索引数组，CSR 格式中每个非零元素的列索引
vector<float> edge_val;             // 边权重数组，CSR 格式中每个非零元素的权重值
vector<int> degree;                 // 顶点度数组，存储每个顶点的度数
vector<int> raw_graph;              // 原始图数据数组，存储每条边的起始顶点和结束顶点

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;   // 浮点数型指针，用于存储矩阵和向量数据

// 读取图数据文件，获取顶点数量、边数量和原始图数据
void readGraph(char *fname) {
  ifstream infile(fname);           // 打开文件流

  int source;
  int end;

  infile >> v_num >> e_num;         // 读取顶点数量和边数量

  while (!infile.eof()) {           // 读取原始图数据
    infile >> source >> end;
    if (infile.peek() == EOF) break;
    raw_graph.push_back(source);
    raw_graph.push_back(end);
  }
}

// 将原始图数据转换为 CSR 格式
void raw_graph_to_CSR() {
  int src;
  int dst;

  row_ptr.resize(v_num + 1, 0);     // 设置行指针数组大小为顶点数量 + 1，并初始化为0
  degree.resize(v_num, 0);          // 设置顶点度数组大小为顶点数量，并初始化为0

  vector<int> temp_degree(v_num, 0);  // 临时顶点度数组，用于存储每个顶点的度数

  #pragma omp parallel for private(src, dst)
  for (int i = 0; i < raw_graph.size() / 2; i++) {  // 遍历原始图数据，计算每个顶点的度数
    src = raw_graph[2 * i];
    dst = raw_graph[2 * i + 1];
    #pragma omp atomic                          // 使用原子操作保证度数的并发更新
    degree[src]++;
  }

  int sum = 0;
  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {               // 计算行指针数组的值
    row_ptr[i] = sum;
    sum += degree[i];
  }
  row_ptr[v_num] = sum;

  col_idx.resize(e_num);                          // 设置列索引数组大小为边数量
  edge_val.resize(e_num);                         // 设置边权重数组大小为边数量

  vector<int> curr_idx(v_num, 0);                  // 当前顶点索引数组，用于记录每个顶点的列索引位置

  #pragma omp parallel for private(src, dst)
  for (int i = 0; i < raw_graph.size() / 2; i++) {  // 遍历原始图数据，填充列索引数组
    src = raw_graph[2 * i];
    dst = raw_graph[2 * i + 1];
    int idx = curr_idx[src]++;
    col_idx[row_ptr[src] + idx] = dst;
  }
}


// 对边进行归一化处理
void edgeNormalization() {
  vector<float> inv_sqrt_degree(v_num);          // 逆度数数组，用于存储每个顶点度数的倒数
  #pragma omp parallel for
  for (int i = 0; i < v_num; i++) {
    inv_sqrt_degree[i] = 1.0 / sqrt(degree[i]);  // 计算每个顶点度数的倒数
  }

  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    for (int j = start; j < end; j++) {           // 遍历每个顶点的邻居节点
      int neighbor = col_idx[j];
      float val = inv_sqrt_degree[i] * inv_sqrt_degree[neighbor];  // 计算边权重
      #pragma omp atomic                         // 使用原子操作进行并发更新
      edge_val[j] += val;
    }
  }
}

// 读取浮点数数据文件
void readFloat(char *fname, float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));     // 分配内存空间
  FILE *fp = fopen(fname, "rb");                   // 以二进制模式打开文件
  fread(dst, num * sizeof(float), 1, fp);          // 读取数据到指针所指向的内存空间
  fclose(fp);                                      // 关闭文件
}

// 初始化浮点数数组为0
void initFloat(float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));     // 分配内存空间
  memset(dst, 0, num * sizeof(float));             // 将内存空间的值设置为0
}

// 矩阵乘法运算
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W) {
  float alpha = 1.0;
  float beta = 0.0;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, v_num, out_dim, in_dim, alpha, in_X, in_dim, W, out_dim, beta, out_X, out_dim);
}

// 矩阵-向量乘法运算
void AX(int dim, float *in_X, float *out_X) {
  float (*tmp_in_X)[dim] = (float(*)[dim])in_X;   // 将输入向量转换为二维数组形式
  float (*tmp_out_X)[dim] = (float(*)[dim])out_X; // 将输出向量转换为二维数组形式

  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < dim; j++) {
      tmp_out_X[i][j] = 0.0;                      // 清零输出向量
    }
  }

  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    for (int j = start; j < end; j++) {           // 遍历每个顶点的邻居节点
      int nbr = col_idx[j];
      float val = edge_val[j];
      float *nbr_vector = tmp_in_X[nbr];
      if (val != 0) {
        cblas_saxpy(dim, val, nbr_vector, 1, tmp_out_X[i], 1);  // 进行向量累加运算
      }
    }
  }
}

// ReLU 激活函数
void ReLU(int dim, float *X) {
  __m256 zero = _mm256_set1_ps(0.0);              // 创建全0的向量
  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num * dim; i += 8) {
    __m256 x = _mm256_loadu_ps(&X[i]);             // 加载待处理的向量数据
    __m256 result = _mm256_max_ps(x, zero);        // 执行 ReLU 激活函数
    _mm256_storeu_ps(&X[i], result);               // 存储处理后的向量数据
  }
}

// LogSoftmax 激活函数
void LogSoftmax(int dim, float* X) {
  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {
    float max_val = -__FLT_MAX__;
    #pragma omp simd reduction(max: max_val)
    for (int j = 0; j < dim; j++) {
      float x = X[i * dim + j];
      if (x > max_val) max_val = x;                // 计算最大值
    }

    float sum = 0.0;
    #pragma omp simd reduction(+: sum)
    for (int j = 0; j < dim; j++) {
      float x = X[i * dim + j];
      X[i * dim + j] = std::exp(x - max_val);      // 计算指数
      sum += X[i * dim + j];                       // 求和
    }
    
    float log_sum = std::log(sum);                  // 计算对数和
    #pragma omp simd
    for (int j = 0; j < dim; j++) {
      X[i * dim + j] = std::log(X[i * dim + j]) - log_sum;  // 计算对数概率
    }
  }
}

// 计算矩阵每行元素的和的最大值
float MaxRowSum(float *X, int dim) {
  float(*tmp_X)[dim] = (float(*)[dim])X;          // 将输入矩阵转换为二维数组形式
  float max = -__FLT_MAX__;

  #pragma omp parallel for reduction(max:max)    // 并行化计算最大值
  for (int i = 0; i < v_num; i++) {
    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += tmp_X[i][j];                          // 计算每行元素的和
    }
    if (sum > max) max = sum;                      // 更新最大值
  }
  return max;
}

// 释放浮点数型指针所分配的内存
void freeFloats() {
  free(X0);
  free(W1);
  free(W2);
  free(X1);
  free(X2);
  free(X1_inter);
  free(X2_inter);
}

// 预处理函数，将原始图数据转换为 CSR 格式
void somePreprocessing() {
  raw_graph_to_CSR();
}

int main(int argc, char **argv) {
  // Do NOT count the time of reading files, malloc, and memset
  F0 = atoi(argv[1]);                            // 获取特征维度参数
  F1 = atoi(argv[2]);
  F2 = atoi(argv[3]);

  readGraph(argv[4]);                             // 读取图数据
  readFloat(argv[5], X0, v_num * F0);              // 读取矩阵和向量数据
  readFloat(argv[6], W1, F0 * F1);
  readFloat(argv[7], W2, F1 * F2);

  initFloat(X1, v_num * F1);                       // 初始化矩阵和向量数据
  initFloat(X1_inter, v_num * F1);
  initFloat(X2, v_num * F2);
  initFloat(X2_inter, v_num * F2);

  // Time point at the start of the computation
  TimePoint start = chrono::steady_clock::now();   // 记录计算开始时间

  somePreprocessing();                             // 预处理，转换原始图数据为 CSR 格式

  edgeNormalization();                             // 对边进行归一化处理

  XW(F0, F1, X0, X1_inter, W1);                    // 进行矩阵乘法运算
  AX(F1, X1_inter, X1);                            // 进行矩阵-向量乘法运算
  ReLU(F1, X1);                                    // 执行 ReLU 激活函数
  XW(F1, F2, X1, X2_inter, W2);                    // 进行矩阵乘法运算
  AX(F2, X2_inter, X2);                            // 进行矩阵-向量乘法运算
  LogSoftmax(F2, X2);                              // 执行 LogSoftmax 激活函数

  // You need to compute the max row sum for result verification
  float max_sum = MaxRowSum(X2, F2);                // 计算矩阵每行元素的和的最大值

  // Time point at the end of the computation
  TimePoint end = chrono::steady_clock::now();     // 记录计算结束时间
  chrono::duration<double> l_durationSec = end - start;
  double l_timeMs = l_durationSec.count() * 1e3;
  printf("%.8f\n", max_sum);
  // Finally, the max row sum and the computing time
  printf("%.8lf\n", l_timeMs);                      // 输出最大行和和计算时间

  // Remember to free your allocated memory
  freeFloats();                                    // 释放内存空间
}
