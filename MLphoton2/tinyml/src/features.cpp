
#include <features.hpp>
#include <math.h>
#include <stdlib.h>
Complex *twiddle_rfft = NULL;
CascadedHPF hpf;
int compare_floats(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}
void hpf_init(void) {
    // Initialize your coefficients (replace with your exact Python design values if needed)
    hpf.b0_0 = 1.0f;  hpf.b1_0 = -2.0f; hpf.b2_0 = 1.0f;
    hpf.a1_0 = -1.9f; hpf.a2_0 = 0.95f;
    hpf.x1_0 = 0.0f;  hpf.x2_0 = 0.0f;  hpf.y1_0 = 0.0f; hpf.y2_0 = 0.0f;

    hpf.b0_1 = 1.0f;  hpf.b1_1 = -2.0f; hpf.b2_1 = 1.0f;
    hpf.a1_1 = -1.9f; hpf.a2_1 = 0.95f;
    hpf.x1_1 = 0.0f;  hpf.x2_1 = 0.0f;  hpf.y1_1 = 0.0f; hpf.y2_1 = 0.0f;
}

float hpf_process(float input) {
    // Stage 1
    float vn_0 = input - hpf.a1_0 * hpf.x1_0 - hpf.a2_0 * hpf.x2_0;
    float out_0 = hpf.b0_0 * vn_0 + hpf.b1_0 * hpf.x1_0 + hpf.b2_0 * hpf.x2_0;
    hpf.x2_0 = hpf.x1_0;
    hpf.x1_0 = vn_0;

    // Stage 2
    float vn_1 = out_0 - hpf.a1_1 * hpf.x1_1 - hpf.a2_1 * hpf.x2_1;
    float out_1 = hpf.b0_1 * vn_1 + hpf.b1_1 * hpf.x1_1 + hpf.b2_1 * hpf.x2_1;
    hpf.x2_1 = hpf.x1_1;
    hpf.x1_1 = vn_1;

    return out_1;
}
void fft_init()
{
    twiddle_rfft = (Complex *)malloc((FFT_SIZE / 2) * sizeof(Complex));
    
    if (twiddle_rfft != NULL) {
        for (int i = 0; i < FFT_SIZE / 2; i++)
        {
            twiddle_rfft[i].real = cosf(-2.0f * (float)M_PI * i / FFT_SIZE);
            twiddle_rfft[i].imag = sinf(-2.0f * (float)M_PI * i / FFT_SIZE);
        }
    }
}

void bit_reversal_shuffle(Complex *data, int n)
{
    int j = 0;
    for (int i = 0; i < n; i++)
    {
        if (i < j)
        {
            // Swap data[i] and data[j]
            Complex temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
        int m = n >> 1;
        while (m >= 1 && j >= m)
        {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}
void cfft_core(Complex *data, int n)
{
    bit_reversal_shuffle(data, n);
    int log2n = log2(n);

    for (int stage = 1; stage <= log2n; stage++)
    {
        int m = 1 << stage;
        int m2 = m >> 1;
        int twiddle_step = n / m;

        for (int j = 0; j < m2; j++)
        {
            Complex w = twiddle_rfft[j * twiddle_step];

            for (int k = j; k < n; k += m)
            {
                int match = k + m2;
                float t_real = data[match].real * w.real - data[match].imag * w.imag;
                float t_imag = data[match].real * w.imag + data[match].imag * w.real;

                data[match].real = data[k].real - t_real;
                data[match].imag = data[k].imag - t_imag;
                data[k].real += t_real;
                data[k].imag += t_imag;
            }
        }
    }
}

void rfft_split(Complex *data, int n)
{
    int count = n / 2;

    // Handle the DC and Nyquist bins
    float r0 = data[0].real;
    float i0 = data[0].imag;
    data[0].real = r0 + i0; // DC component
    data[0].imag = r0 - i0; // Nyquist component (stored in data[0].imag)

    for (int i = 1; i <= count / 2; i++)
    {
        int j = count - i;

        Complex t1 = data[i];
        Complex t2 = data[j];

        // Separate the two interleaved spectra
        Complex even, odd;
        even.real = 0.5f * (t1.real + t2.real);
        even.imag = 0.5f * (t1.imag - t2.imag);

        odd.real = 0.5f * (t1.imag + t2.imag);
        odd.imag = -0.5f * (t1.real - t2.real);

        // Multiply 'odd' by the RFFT twiddle factors
        Complex w = twiddle_rfft[i];
        float r_twid = odd.real * w.real - odd.imag * w.imag;
        float i_twid = odd.real * w.imag + odd.imag * w.real;

        // Reconstruct the actual frequency spectrum
        data[i].real = even.real + r_twid;
        data[i].imag = even.imag + i_twid;

        data[j].real = even.real - r_twid;
        data[j].imag = -even.imag + i_twid;
    }
}

TimeFeatures extract_time_features(float *clip, int len)
{
    TimeFeatures tf;

    // 1. Min, Max, Mean, RMS, and ZCR (Single Pass)
    float sum = 0.0f;
    float abs_sum = 0.0f;
    float sq_sum = 0.0f;
    tf.min = clip[0];
    tf.max = clip[0];
    int zero_crossings = 0;
    int peaks = 0;

    for (int i = 0; i < len; i++)
    {
        sum += clip[i];
        abs_sum += fabsf(clip[i]);
        sq_sum += clip[i] * clip[i];

        if (clip[i] < tf.min)
            tf.min = clip[i];
        if (clip[i] > tf.max)
            tf.max = clip[i];

        // Zero Crossing Rate calculation
        if (i > 0)
        {
            // Check if sign changed
            if ((clip[i] >= 0 && clip[i - 1] < 0) || (clip[i] < 0 && clip[i - 1] >= 0))
            {
                zero_crossings++;
            }

            // Simple Peak Find (local maxima)
            if (i < len - 1 && clip[i] > clip[i - 1] && clip[i] > clip[i + 1] && clip[i] > 0.0f)
            {
                peaks++;
            }
        }
    }

    tf.mean = abs_sum / len;
    float true_mean = sum / len;
    tf.rms = sqrtf(sq_sum / len);
    tf.ptp = tf.max - tf.min;
    tf.max_mean = tf.max / tf.mean;
    tf.zcr = (float)zero_crossings / (len - 1); // normalized ZCR
    tf.peak_count = (float)peaks;

    // 2. Variance, Standard Deviation, Skewness, and Kurtosis (Second Pass)
    float var_sum = 0.0f;
    float skew_sum = 0.0f;
    float kurt_sum = 0.0f;
    for (int i = 0; i < len; i++)
    {
        float diff = clip[i] - true_mean;
        float diff_sq = diff * diff;
        var_sum += diff_sq;
        skew_sum += diff_sq * diff;
        kurt_sum += diff_sq * diff_sq;
    }
    float variance = var_sum / len;
    tf.std = sqrtf(variance);

    // Skewness and Kurtosis standard formulations
    tf.skew = (skew_sum / len) / powf(tf.std, 3.0f);
    tf.kurtosis = ((kurt_sum / len) / powf(tf.std, 4.0f)) - 3.0f; // excess kurtosis

    // 3. Median, IQR, and MAD (Requires Sorting)
    // Create a temporary static copy of the array so we don't mess up the original or overflow stack
    static float sorted_clip[8192];
    for (int i = 0; i < len; i++)
        sorted_clip[i] = clip[i];
    qsort(sorted_clip, len, sizeof(float), compare_floats);

    float median = sorted_clip[len / 2];
    tf.iqr = sorted_clip[(int)(len * 0.75f)] - sorted_clip[(int)(len * 0.25f)];

    // Calculate MAD: median(abs(clip - median))
    static float mad_array[8192];
    for (int i = 0; i < len; i++)
    {
        mad_array[i] = fabsf(clip[i] - median);
    }
    qsort(mad_array, len, sizeof(float), compare_floats);
    tf.mad = mad_array[len / 2];

    return tf;
}

int process_signal_for_emlearn(float *my_raw_data)
{
    // Allocate the complex buffer on the stack (grows/shrinks dynamically)
    // or keep static if stack size is constrained, but let's drop filtered_audio!
    Complex *fft_buffer = (Complex *)malloc(CFFT_SIZE * sizeof(Complex));
    float *master_feature_vector = (float *)malloc(TOTAL_FEATURES_COUNT * sizeof(float));
    // 1. In-place HPF processing to avoid creating a massive 'filtered_audio' array
    for (int i = 0; i < CFFT_SIZE; i++)
    {
        int even_idx = 2 * i;
        int odd_idx = 2 * i + 1;

        if (even_idx < ACTUAL_AUDIO_SAMPLES)
        {
            // Modify my_raw_data directly in place!
            my_raw_data[even_idx] = hpf_process(my_raw_data[even_idx]);
            fft_buffer[i].real = my_raw_data[even_idx];
        }
        else
        {
            fft_buffer[i].real = 0.0f;
        }

        if (odd_idx < ACTUAL_AUDIO_SAMPLES)
        {
            my_raw_data[odd_idx] = hpf_process(my_raw_data[odd_idx]);
            fft_buffer[i].imag = my_raw_data[odd_idx];
        }
        else
        {
            fft_buffer[i].imag = 0.0f;
        }
    }

    // 2. Run the RFFT pipeline
    cfft_core(fft_buffer, CFFT_SIZE);
    rfft_split(fft_buffer, CFFT_SIZE);

    // 3. Extract Frequency magnitudes
    master_feature_vector[0] = fabsf(fft_buffer[0].real);
    master_feature_vector[CFFT_SIZE] = fabsf(fft_buffer[0].imag);

    for (int i = 1; i < CFFT_SIZE; i++)
    {
        float r = fft_buffer[i].real;
        float j = fft_buffer[i].imag;
        master_feature_vector[i] = sqrtf(r * r + j * j);
    }

    // 4. Pass the already-filtered raw_data array directly to time features!
    // This saves us an entire 32KB array allocation.
    TimeFeatures tf = extract_time_features(my_raw_data, ACTUAL_AUDIO_SAMPLES);

    // 5. Populate feature vector (Index 4097+)
    int idx = 4097;
    master_feature_vector[idx++] = tf.mean;
    master_feature_vector[idx++] = tf.std;
    master_feature_vector[idx++] = tf.min;
    master_feature_vector[idx++] = tf.max;
    master_feature_vector[idx++] = tf.ptp;
    master_feature_vector[idx++] = tf.mad;
    master_feature_vector[idx++] = tf.iqr;
    master_feature_vector[idx++] = tf.max_mean;
    master_feature_vector[idx++] = tf.rms;
    master_feature_vector[idx++] = tf.zcr;
    master_feature_vector[idx++] = tf.peak_count;
    master_feature_vector[idx++] = tf.skew;
    master_feature_vector[idx++] = tf.kurtosis;

    // 6. ML Prediction Logic
    float confidence_threshold = 0.70f;
    float probabilities[4] = {0.0f};
    int class_id; // = my_emlearn_model_predict_proba(master_feature_vector, TOTAL_FEATURES_COUNT, probabilities, 4);
    free(fft_buffer);
    free(master_feature_vector);
    if (probabilities[class_id] >= confidence_threshold)
    {
        if (class_id >= 0 && class_id < 4)
            return class_id;
    }
    return -1;
}