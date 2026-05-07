/*
 * Project myProject
 * Author: Your Name
 * Date:
 * For comprehensive documentation and examples, please visit:
 * https://docs.particle.io/firmware/best-practices/firmware-template/
 */

// Include Particle Device OS APIs
// #include "Particle.h"

#ifndef __FPU_PRESENT
#define __FPU_PRESENT 1
#endif
#include "arm_math.h"
#include "Microphone_PDM.h"

#define FFT_SIZE 8192 // POWER OF 2!!! 16KHZ/500MS = 8K SAMPLES 0 PAD REST
int samples_collected = 0;
unsigned long recordingStart;

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

// FFT instance
//arm_rfft_instance_q15 S;
arm_rfft_fast_instance_f32 S;

int16_t mic_buffer[FFT_SIZE];
float32_t fft_input[FFT_SIZE];     // The "f32" version for math
float32_t fft_output[FFT_SIZE * 2]; // Result buffer (Complex: Real + Imag)
float32_t mag_buf[FFT_SIZE / 2];
// If you don't hit the setup button to stop recording, this is how long to go before turning it
// off automatically. The limit really is only the disk space available to receive the file.
const unsigned long MAX_RECORDING_LENGTH_MS = 30000;

enum State
{
  STATE_WAITING,
  STATE_CONNECT,
  STATE_RUNNING,
  STATE_FINISH
};
State state = STATE_WAITING;
// setup() runs once, when the device is first turned on
void setup()
{

  Particle.connect();
  Serial.begin(); // Ensure Serial is initialized
                  // Register handler to handle clicking on the SETUP button

  // spawn fft
  arm_rfft_fast_init_f32(&S, FFT_SIZE);

  // buffer for audio sampled
  // PDM Mic instance

  int err = Microphone_PDM::instance()
                .withOutputSize(Microphone_PDM::OutputSize::SIGNED_16)
                .withRange(Microphone_PDM::Range::RANGE_2048) // 2048
                .withSampleRate(16000)                        // 16000
                .init();

  if (err)
  {
    Log.error("PDM decoder init err=%d", err);
  }

  err = Microphone_PDM::instance().start();
  if (err)
  {
    Log.error("PDM decoder start err=%d", err);
  }
}

// loop() runs over and over again, as quickly as it can execute.
void loop()
{

  switch (state)
  {
  case STATE_WAITING:
    /* code */
    break;

  case STATE_RUNNING:

    recordingStart = millis();

    Microphone_PDM::instance().noCopySamples([](void *pSamples, size_t numSamples)
                                             {int16_t *incomingData = (int16_t *)pSamples;
                                            size_t count = Microphone_PDM::instance().getBufferSizeInBytes() / 2;

    for (size_t i = 0; i < count; i++) {
        if (samples_collected < FFT_SIZE) {
            mic_buffer[samples_collected] = incomingData[i];
            samples_collected++;
        }
      } });

      arm_q15_to_float((q15_t*)mic_buffer,fft_input,0)
    if (samples_collected == FFT_SIZE)
    {
      arm_rfft_fast_f32(&S, fft_input, fft_output, 0);
      arm_cmplx_mag_f32(fft_output, mag_buf, FFT_SIZE / 2);
      
    }

    Serial.println("500 MS sound samples With FFT 1024 F32");
    Serial.println(millis() - recordingStart);
    if (millis() - recordingStart >= MAX_RECORDING_LENGTH_MS)
    {
      state = STATE_FINISH;
    }
    break;
  case STATE_FINISH:

    Log.info("stopping");
    state = STATE_WAITING;
    break;

  default:
    break;
  }

  // Example: Publish event to cloud every 10 seconds. Uncomment the next 3 lines to try it!
  // Log.info("Sending Hello World to the cloud!");
  // Particle.publish("Hello world!");
  // delay( 10 * 1000 ); // milliseconds and blocking - see docs for more info!
}
