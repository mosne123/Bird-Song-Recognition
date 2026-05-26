#ifndef FEATURES_HPP
#define FEATURES_HPP

#include <math.h>
#define ACTUAL_AUDIO_SAMPLES 7800          // 16kHz sr * 0.5s
#define FFT_SIZE 8192                      // POWER OF 2
#define CFFT_SIZE (FFT_SIZE / 2)           // 4096
#define FFT_FEATURES_COUNT (CFFT_SIZE + 1) // 4097
#define TIME_FEATURES_COUNT 13
#define TOTAL_FEATURES_COUNT (FFT_FEATURES_COUNT + TIME_FEATURES_COUNT) // 4110

typedef struct
{
    float real;
    float imag;
} Complex;

typedef struct
{
    float mean, std, min, max, ptp, mad, iqr, max_mean, rms, zcr, peak_count, skew, kurtosis;
} TimeFeatures;

typedef struct
{
    // Stage 1
    float b0_0, b1_0, b2_0, a1_0, a2_0;
    float x1_0, x2_0, y1_0, y2_0;
    // Stage 2
    float b0_1, b1_1, b2_1, a1_1, a2_1;
    float x1_1, x2_1, y1_1, y2_1;
} CascadedHPF;

void hpf_init(void);

float hpf_process(float input);

void cfft_core(Complex *data, int n);
void fft_init(void);
void rfft_split(Complex *data, int n, Complex *twiddle_rfft);
void bit_reversal_shuffle(Complex *data, int n);
int compare_floats(const void *a, const void *b);
TimeFeatures extract_time_features(float *clip, int len);
int process_signal_for_emlearn(float *my_raw_data);

#endif