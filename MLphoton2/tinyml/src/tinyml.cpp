/*
 * Project myProject
 * Author: Your Name
 * Date:
 * For comprehensive documentation and examples, please visit:
 * https://docs.particle.io/firmware/best-practices/firmware-template/
 */

// Include Particle Device OS APIs

#ifndef __FPU_PRESENT
#define __FPU_PRESENT 1
#endif

#include "Microphone_PDM.h"
#include "features.hpp"
// #include "Particle.h"

int samples_collected = 0;
unsigned long recordingStart;

const char *class_names[] = {
    "Graaspurv",  // Class 0
    "Haettemaage",  // Class 1
    "Krage", // Class 2
    "Raage",   // Class 3
    "Solsort",
    "Unknown"

};

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);
// FFT instance
// arm_rfft_instance_q15 S;
typedef float float32_t;
int16_t mic_buffer[ACTUAL_AUDIO_SAMPLES];
int class_id;
// If you don't hit the setup button to stop recording, this is how long to go before turning it
// off automatically. The limit really is only the disk space available to receive the file.
const unsigned long MAX_RECORDING_LENGTH_MS = 500;

enum State
{
  STATE_WAITING,
  STATE_CONNECT,
  STATE_RUNNING,
  STATE_FINISH
};
State state = STATE_RUNNING;

bool buffer_ready = false;
void setup()
{
  hpf_init();
  fft_init();
  uint32_t mem = System.freeMemory();
  Particle.connect();
  Serial.begin(); // Ensure Serial is initialized
                  // Register handler to handle clicking on the SETUP button
  //
  Serial.println("Free memory at start up");
  Serial.println(mem);

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
    {
    recordingStart = millis();

    Microphone_PDM::instance().noCopySamples([](void *pSamples, size_t numSamples)
                                             {
    int16_t *incomingData = (int16_t *)pSamples;

    // USE numSamples INSTEAD OF WALKING THE WHOLE HARDWARE BUFFER
    for (size_t i = 0; i < numSamples; i++) 
    {
        if (!buffer_ready && samples_collected < ACTUAL_AUDIO_SAMPLES) 
        { 
            mic_buffer[samples_collected] = incomingData[i];
            samples_collected++;

            if (samples_collected >= ACTUAL_AUDIO_SAMPLES) 
            {
                buffer_ready = true; // Lock buffer and signal the main loop
            }
          }
        }
    } );

    if (buffer_ready)
    {
      float *float_audio_clip = (float *)malloc(ACTUAL_AUDIO_SAMPLES * sizeof(float));
      // Convert raw int16_t audio (-32768 to 32767) to normalized float (-1.0 to 1.0)
      for (int i = 0; i < ACTUAL_AUDIO_SAMPLES; i++)
      {
        float_audio_clip[i] = (float)mic_buffer[i] / 32768.0f;
      }

      // Reset tracking variables to start recording the next 500ms window
      samples_collected = 0;
      buffer_ready = false;

      class_id = process_signal_for_emlearn(float_audio_clip);
      if (class_id >= 0)
      {
        Serial.printf("Confirmed Detection: %s\n", class_names[class_id]);
      }
      //Serial.printf("Detected Class ID: %d\n", class_id);
      //Serial.println("500 MS sound samples With FFT 8196 Float");
     // Serial.println(millis() - recordingStart);
      free(float_audio_clip);
      if (millis() - recordingStart >= MAX_RECORDING_LENGTH_MS)
      {
        state = STATE_FINISH;
      }
     
      }
       break;
    case STATE_FINISH:

      //Log.info("stopping");
      state = STATE_RUNNING;
      break;

    default:
      break;
    }
  }
}