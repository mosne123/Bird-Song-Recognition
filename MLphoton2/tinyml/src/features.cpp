#define EMLEARN_FEATURES_FLOAT 1

#include "Particle.h"
#include "features.hpp"
#include <math.h>
#include <stdlib.h>

// Make sure this name exactly matches your generated emlearn header file!
#include "Bird_recog_model.h" 

CascadedHPF hpf;
Complex *twiddle_rfft = NULL;

int compare_floats(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

void hpf_init(void) {
    hpf.b0_0 = 1.0f;  hpf.b1_0 = -2.0f; hpf.b2_0 = 1.0f;
    hpf.a1_0 = -1.9f; hpf.a2_0 = 0.95f;
    hpf.x1_0 = 0.0f;  hpf.x2_0 = 0.0f;  hpf.y1_0 = 0.0f; hpf.y2_0 = 0.0f;

    hpf.b0_1 = 1.0f;  hpf.b1_1 = -2.0f; hpf.b2_1 = 1.0f;
    hpf.a1_1 = -1.9f; hpf.a2_1 = 0.95f;
    hpf.x1_1 = 0.0f;  hpf.x2_1 = 0.0f;  hpf.y1_1 = 0.0f; hpf.y2_1 = 0.0f;
}

float hpf_process(float input) {
    float vn_0 = input - hpf.a1_0 * hpf.x1_0 - hpf.a2_0 * hpf.x2_0;
    float out_0 = hpf.b0_0 * vn_0 + hpf.b1_0 * hpf.x1_0 + hpf.b2_0 * hpf.x2_0;
    hpf.x2_0 = hpf.x1_0;
    hpf.x1_0 = vn_0;

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

    float r0 = data[0].real;
    float i0 = data[0].imag;
    data[0].real = r0 + i0;
    data[0].imag = r0 - i0;

    for (int i = 1; i <= count / 2; i++)
    {
        int j = count - i;

        Complex t1 = data[i];
        Complex t2 = data[j];

        Complex even, odd;
        even.real = 0.5f * (t1.real + t2.real);
        even.imag = 0.5f * (t1.imag - t2.imag);

        odd.real = 0.5f * (t1.imag + t2.imag);
        odd.imag = -0.5f * (t1.real - t2.real);

        Complex w = twiddle_rfft[i];
        float r_twid = odd.real * w.real - odd.imag * w.imag;
        float i_twid = odd.real * w.imag + odd.imag * w.real;

        data[i].real = even.real + r_twid;
        data[i].imag = even.imag + i_twid;

        data[j].real = even.real - r_twid;
        data[j].imag = -even.imag + i_twid;
    }
}

// Fixed to only extract features active in Python script
TimeFeatures extract_time_features(float *clip, int len)
{
    TimeFeatures tf;
    float abs_sum = 0.0f;
    int peaks = 0;

    for (int i = 0; i < len; i++)
    {
        abs_sum += fabsf(clip[i]);

        if (i > 0 && i < len - 1)
        {
            if (clip[i] > clip[i - 1] && clip[i] > clip[i + 1])
            {
                peaks++;
            }
        }
    }

    tf.mean = abs_sum / len; // Equivalent to Python's np.mean(abs(clip))
    tf.peak_count = (float)peaks;

    float *sorted_clip = (float *)malloc(len * sizeof(float));
    if (sorted_clip != NULL) {
        for (int i = 0; i < len; i++) sorted_clip[i] = clip[i];
        qsort(sorted_clip, len, sizeof(float), compare_floats);

        // Explicitly matches Python's np.percentile bounds
        float p75 = sorted_clip[(int)(len * 0.75f)];
        float p25 = sorted_clip[(int)(len * 0.25f)];
        tf.iqr = p75 - p25;
        
        free(sorted_clip);
    } else {
        tf.iqr = 0.0f;
    }

    return tf;
}

int process_signal_for_emlearn(float *my_raw_data)
{
    Complex *fft_buffer = (Complex *)malloc(CFFT_SIZE * sizeof(Complex));
    float *magnitude_spectrum = (float *)malloc(CFFT_SIZE * sizeof(float));
    float *master_feature_vector = (float *)malloc(TOTAL_FEATURES_COUNT * sizeof(float));
    
    if (!fft_buffer || !magnitude_spectrum || !master_feature_vector) {
        if (fft_buffer) free(fft_buffer);
        if (magnitude_spectrum) free(magnitude_spectrum);
        if (master_feature_vector) free(master_feature_vector);
        return -1;
    }

    // 1. Audio clip level normalization to match Python preprocessing
    float max_val = 0.0f;
    for (int i = 0; i < ACTUAL_AUDIO_SAMPLES; i++) {
        float abs_v = fabsf(my_raw_data[i]);
        if (abs_v > max_val) max_val = abs_v;
    }
    if (max_val > 0.0f) {
        for (int i = 0; i < ACTUAL_AUDIO_SAMPLES; i++) {
            my_raw_data[i] = my_raw_data[i] / max_val;
        }
    }

    // 2. High-Pass Filtering & FFT Buffer Packing
    for (int i = 0; i < CFFT_SIZE; i++)
    {
        int even_idx = 2 * i;
        int odd_idx = 2 * i + 1;

        if (even_idx < ACTUAL_AUDIO_SAMPLES)
        {
            my_raw_data[even_idx] = hpf_process(my_raw_data[even_idx]);
            fft_buffer[i].real = my_raw_data[even_idx];
        }
        else fft_buffer[i].real = 0.0f;

        if (odd_idx < ACTUAL_AUDIO_SAMPLES)
        {
            my_raw_data[odd_idx] = hpf_process(my_raw_data[odd_idx]);
            fft_buffer[i].imag = my_raw_data[odd_idx];
        }
        else fft_buffer[i].imag = 0.0f;
    }

    cfft_core(fft_buffer, CFFT_SIZE);
    rfft_split(fft_buffer, CFFT_SIZE);

    // 3. Absolute Magnitude Spectrum Generation (Matches Python: np.abs(np.fft.rfft))
    magnitude_spectrum[0] = fabsf(fft_buffer[0].real);
    for (int i = 1; i < CFFT_SIZE; i++)
    {
        float r = fft_buffer[i].real;
        float j = fft_buffer[i].imag;
        magnitude_spectrum[i] = sqrtf(r * r + j * j);
    }

    // 4. Calculate Time Features
    TimeFeatures tf = extract_time_features(my_raw_data, ACTUAL_AUDIO_SAMPLES);

    // 5. Calculate Spectral Centroid
    float sum_magnitude = 0.0f;
    float sum_weighted_freq = 0.0f;
    float bin_resolution = 16000.0f / (float)FFT_SIZE;

    for (int i = 0; i < CFFT_SIZE; i++)
    {
        float freq = (float)i * bin_resolution;
        sum_weighted_freq += freq * magnitude_spectrum[i];
        sum_magnitude += magnitude_spectrum[i];
    }
    float spectral_centroid = (sum_magnitude > 0.0f) ? (sum_weighted_freq / sum_magnitude) : 0.0f;

    // 6. Calculate Spectral Entropy
    float entropy = 0.0f;
    if (sum_magnitude > 0.0f) {
        for (int i = 0; i < CFFT_SIZE; i++) {
            float psd = magnitude_spectrum[i] / sum_magnitude;
            entropy -= psd * log2f(psd + 1e-10f);
        }
    }

    // 7. Synchronize Columns directly to the Master Feature Vector
    
    int idx = 0;
    master_feature_vector[idx++] = tf.mean;          // Column 1: ABS_mean
    master_feature_vector[idx++] = tf.iqr;           // Column 2: IQR
    master_feature_vector[idx++] = tf.peak_count;    // Column 3: peak_count
    master_feature_vector[idx++] = spectral_centroid;// Column 4: spectral_centroid

    // Column 5-12: Band Energies (Using absolute magnitude squared inside bands)
    float bands_def[8][2] = {
        {0, 500}, {500, 1000}, {1000, 1500}, {1500, 2000},
        {2000, 3000}, {3000, 4000}, {4000, 6000}, {6000, 8000}
    };

    for (int b = 0; b < 8; b++) {
        int start_bin = (int)(bands_def[b][0] / bin_resolution);
        int end_bin = (int)(bands_def[b][1] / bin_resolution);
        if (end_bin > CFFT_SIZE) end_bin = CFFT_SIZE;

        float band_energy = 0.0f;
        for (int i = start_bin; i < end_bin; i++) {
            band_energy += magnitude_spectrum[i] * magnitude_spectrum[i];
        }
        master_feature_vector[idx++] = band_energy;
    }

    master_feature_vector[idx++] = entropy;          // Column 13: spectral_entropy

    // 8. Telemetry Inspection Debugging
    Serial.println("\n=== REAL TINYML SYNCED TELEMETRY ===");
    Serial.printf("ABS_mean: %f | IQR: %f | Peaks: %f\n", master_feature_vector[0], master_feature_vector[1], master_feature_vector[2]);
    Serial.printf("Centroid: %f | Entropy: %f\n", master_feature_vector[3], master_feature_vector[12]);
    Serial.printf("Band 0-500Hz Energy: %f\n", master_feature_vector[4]);
    Serial.printf("Band 3k-4kHz Energy: %f\n", master_feature_vector[9]);
    Serial.println("====================================");

    free(fft_buffer);
    free(magnitude_spectrum);

    // 9. ML Execution Logic
    float confidence_threshold = 0.30f;
    float probabilities[6] = {0.0f};
    

    //WARNING!!!! 1. 6 tal er feature længde. men det er da 13 og ikke 6?!?!
    int class_id = Bird_recog_model_predict_proba(master_feature_vector, 13, probabilities, 6);
    free(master_feature_vector); 

   int winner_class_id = 0;
    float max_prob = probabilities[0];
    
    for (int i = 1; i < 6; i++) {
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            winner_class_id = i;
        }
    }

    Serial.print("Class Probabilities -> ");
    for(int i = 0; i < 6; i++) {
        Serial.printf("[%d]: %.4f ", i, probabilities[i]);
    }
    Serial.printf("\nActual Highest Class ID: %d (Prob: %.4f)\n", winner_class_id, max_prob);

    // Threshold evaluation based on the true maximum probability
    if (max_prob >= confidence_threshold) {
        return winner_class_id;
    }
    
    return -1;
}