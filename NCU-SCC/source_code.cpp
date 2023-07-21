#include <math.h>                   // ��ѧ�����⣬������ѧ������صĺ���
#include <omp.h>                    // OpenMP ͷ�ļ���֧�ֲ��м���
#include <stdio.h>                  // ��׼�������������
#include <string.h>                 // �ַ�������������
#include <cblas.h>                  // CBLAS ͷ�ļ������ڵ��� BLAS��Basic Linear Algebra Subprograms�����еĺ���
#include <vector>                   // ���������⣬���ڶ���Ͳ�������
#include <chrono>                   // ʱ��⣬���ڼ����������ʱ��
#include <fstream>                  // �ļ����⣬�����ļ���д����
#include <iostream>                 // �����������
#include <immintrin.h>              // Intel SIMD��Single Instruction, Multiple Data��ָ�ͷ�ļ�������������ָ��
#include <thread>
#include <cstdlib>
using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;     // ʱ�������
auto max_threads=std::thread::hardware_concurrency();
int v_num = 0;                      // ��������
int e_num = 0;                      // ������
int F0 = 0, F1 = 0, F2 = 0;         // ����ά��

vector<int> row_ptr;                // ��ָ�����飬CSR ��ʽ��ÿһ����ʼλ�õ�����
vector<int> col_idx;                // ���������飬CSR ��ʽ��ÿ������Ԫ�ص�������
vector<float> edge_val;             // ��Ȩ�����飬CSR ��ʽ��ÿ������Ԫ�ص�Ȩ��ֵ
vector<int> degree;                 // ��������飬�洢ÿ������Ķ���
vector<int> raw_graph;              // ԭʼͼ�������飬�洢ÿ���ߵ���ʼ����ͽ�������

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;   // ��������ָ�룬���ڴ洢�������������

// ��ȡͼ�����ļ�����ȡ������������������ԭʼͼ����
void readGraph(char *fname) {
  ifstream infile(fname);           // ���ļ���

  int source;
  int end;

  infile >> v_num >> e_num;         // ��ȡ���������ͱ�����

  while (!infile.eof()) {           // ��ȡԭʼͼ����
    infile >> source >> end;
    if (infile.peek() == EOF) break;
    raw_graph.push_back(source);
    raw_graph.push_back(end);
  }
}

// ��ԭʼͼ����ת��Ϊ CSR ��ʽ
void raw_graph_to_CSR() {
  int src;
  int dst;

  row_ptr.resize(v_num + 1, 0);     // ������ָ�������СΪ�������� + 1������ʼ��Ϊ0
  degree.resize(v_num, 0);          // ���ö���������СΪ��������������ʼ��Ϊ0

  vector<int> temp_degree(v_num, 0);  // ��ʱ��������飬���ڴ洢ÿ������Ķ���

  #pragma omp parallel for private(src, dst)
  for (int i = 0; i < raw_graph.size() / 2; i++) {  // ����ԭʼͼ���ݣ�����ÿ������Ķ���
    src = raw_graph[2 * i];
    dst = raw_graph[2 * i + 1];
    #pragma omp atomic                          // ʹ��ԭ�Ӳ�����֤�����Ĳ�������
    degree[src]++;
  }

  int sum = 0;
  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {               // ������ָ�������ֵ
    row_ptr[i] = sum;
    sum += degree[i];
  }
  row_ptr[v_num] = sum;

  col_idx.resize(e_num);                          // ���������������СΪ������
  edge_val.resize(e_num);                         // ���ñ�Ȩ�������СΪ������

  vector<int> curr_idx(v_num, 0);                  // ��ǰ�����������飬���ڼ�¼ÿ�������������λ��

  #pragma omp parallel for private(src, dst)
  for (int i = 0; i < raw_graph.size() / 2; i++) {  // ����ԭʼͼ���ݣ��������������
    src = raw_graph[2 * i];
    dst = raw_graph[2 * i + 1];
    int idx = curr_idx[src]++;
    col_idx[row_ptr[src] + idx] = dst;
  }
}


// �Ա߽��й�һ������
void edgeNormalization() {
  vector<float> inv_sqrt_degree(v_num);          // ��������飬���ڴ洢ÿ����������ĵ���
  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {
    inv_sqrt_degree[i] = 1.0 / sqrt(degree[i]);  // ����ÿ����������ĵ���
  }

  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    for (int j = start; j < end; j++) {           // ����ÿ��������ھӽڵ�
      int neighbor = col_idx[j];
      float val = inv_sqrt_degree[i] * inv_sqrt_degree[neighbor];  // �����Ȩ��
      #pragma omp atomic                         // ʹ��ԭ�Ӳ������в�������
      edge_val[j] += val;
    }
  }
}

// ��ȡ�����������ļ�
void readFloat(char *fname, float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));     // �����ڴ�ռ�
  FILE *fp = fopen(fname, "rb");                   // �Զ�����ģʽ���ļ�
  fread(dst, num * sizeof(float), 1, fp);          // ��ȡ���ݵ�ָ����ָ����ڴ�ռ�
  fclose(fp);                                      // �ر��ļ�
}

// ��ʼ������������Ϊ0
void initFloat(float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));     // �����ڴ�ռ�
  memset(dst, 0, num * sizeof(float));             // ���ڴ�ռ��ֵ����Ϊ0
}

// ����˷�����
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W) {
  float alpha = 1.0;
  float beta = 0.0;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, v_num, out_dim, in_dim, alpha, in_X, in_dim, W, out_dim, beta, out_X, out_dim);
}

// ����-�����˷�����
void AX(int dim, float *in_X, float *out_X) {
  float (*tmp_in_X)[dim] = (float(*)[dim])in_X;   // ����������ת��Ϊ��ά������ʽ
  float (*tmp_out_X)[dim] = (float(*)[dim])out_X; // ���������ת��Ϊ��ά������ʽ

  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < dim; j++) {
      tmp_out_X[i][j] = 0.0;                      // �����������
    }
  }

  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    for (int j = start; j < end; j++) {           // ����ÿ��������ھӽڵ�
      int nbr = col_idx[j];
      float val = edge_val[j];
      float *nbr_vector = tmp_in_X[nbr];
      if (val != 0) {
        cblas_saxpy(dim, val, nbr_vector, 1, tmp_out_X[i], 1);  // ���������ۼ�����
      }
    }
  }
}

// ReLU �����
void ReLU(int dim, float *X) {
  __m256 zero = _mm256_set1_ps(0.0);              // ����ȫ0������
  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num * dim; i += 8) {
    __m256 x = _mm256_loadu_ps(&X[i]);             // ���ش���������������
    __m256 result = _mm256_max_ps(x, zero);        // ִ�� ReLU �����
    _mm256_storeu_ps(&X[i], result);               // �洢���������������
  }
}

// LogSoftmax �����
void LogSoftmax(int dim, float* X) {
  #pragma omp parallel for num_threads(max_threads-1)
  for (int i = 0; i < v_num; i++) {
    float max_val = -__FLT_MAX__;
    #pragma omp simd reduction(max: max_val)
    for (int j = 0; j < dim; j++) {
      float x = X[i * dim + j];
      if (x > max_val) max_val = x;                // �������ֵ
    }

    float sum = 0.0;
    #pragma omp simd reduction(+: sum)
    for (int j = 0; j < dim; j++) {
      float x = X[i * dim + j];
      X[i * dim + j] = std::exp(x - max_val);      // ����ָ��
      sum += X[i * dim + j];                       // ���
    }

    float log_sum = std::log(sum);                  // ���������
    #pragma omp simd
    for (int j = 0; j < dim; j++) {
      X[i * dim + j] = std::log(X[i * dim + j]) - log_sum;  // �����������
    }
  }
}

// �������ÿ��Ԫ�صĺ͵����ֵ
float MaxRowSum(float *X, int dim) {
  float(*tmp_X)[dim] = (float(*)[dim])X;          // ���������ת��Ϊ��ά������ʽ
  float max = -__FLT_MAX__;

  #pragma omp parallel for reduction(max:max)    // ���л��������ֵ
  for (int i = 0; i < v_num; i++) {
    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += tmp_X[i][j];                          // ����ÿ��Ԫ�صĺ�
    }
    if (sum > max) max = sum;                      // �������ֵ
  }
  return max;
}

// �ͷŸ�������ָ����������ڴ�
void freeFloats() {
  free(X0);
  free(W1);
  free(W2);
  free(X1);
  free(X2);
  free(X1_inter);
  free(X2_inter);
}

// Ԥ������������ԭʼͼ����ת��Ϊ CSR ��ʽ
void somePreprocessing() {
  raw_graph_to_CSR();
}

int main(int argc, char **argv) {
  // Do NOT count the time of reading files, malloc, and memset
  F0 = atoi(argv[1]);                            // ��ȡ����ά�Ȳ���
  F1 = atoi(argv[2]);
  F2 = atoi(argv[3]);

  readGraph(argv[4]);                             // ��ȡͼ����
  readFloat(argv[5], X0, v_num * F0);              // ��ȡ�������������
  readFloat(argv[6], W1, F0 * F1);
  readFloat(argv[7], W2, F1 * F2);

  initFloat(X1, v_num * F1);                       // ��ʼ���������������
  initFloat(X1_inter, v_num * F1);
  initFloat(X2, v_num * F2);
  initFloat(X2_inter, v_num * F2);

  // Time point at the start of the computation
  TimePoint start = chrono::steady_clock::now();   // ��¼���㿪ʼʱ��

  somePreprocessing();                             // Ԥ������ת��ԭʼͼ����Ϊ CSR ��ʽ

  edgeNormalization();                             // �Ա߽��й�һ������

  XW(F0, F1, X0, X1_inter, W1);                    // ���о���˷�����
  AX(F1, X1_inter, X1);                            // ���о���-�����˷�����
  ReLU(F1, X1);                                    // ִ�� ReLU �����
  XW(F1, F2, X1, X2_inter, W2);                    // ���о���˷�����
  AX(F2, X2_inter, X2);                            // ���о���-�����˷�����
  LogSoftmax(F2, X2);                              // ִ�� LogSoftmax �����

  // You need to compute the max row sum for result verification
  float max_sum = MaxRowSum(X2, F2);                // �������ÿ��Ԫ�صĺ͵����ֵ

  // Time point at the end of the computation
  TimePoint end = chrono::steady_clock::now();     // ��¼�������ʱ��
  chrono::duration<double> l_durationSec = end - start;
  double l_timeMs = l_durationSec.count() * 1e3;
  // printf("%.8f\n", max_sum);
  // Finally, the max row sum and the computing time
  printf("%.8lf\n", l_timeMs);                      // �������кͺͼ���ʱ��

  // Remember to free your allocated memory
  freeFloats();                                    // �ͷ��ڴ�ռ�
}
