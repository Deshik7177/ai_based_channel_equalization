/*
 * ESP32 Signal Generator for Channel Equalization Project
 * Generates a clean signal: Sine wave + Random binary data
 * Sends data via Serial at 115200 baud.
 */

#include <Arduino.h>

const float freq = 1.0; // Frequency of sine wave in Hz
const float sampling_rate = 50.0; // Samples per second
const unsigned long interval = 1000 / sampling_rate;

unsigned long lastTime = 0;
float t = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial); // Wait for Serial to be ready
}

void loop() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastTime >= interval) {
    lastTime = currentTime;
    
    // Generate Sine Wave component
    float sine_val = sin(2 * PI * freq * t);
    
    // Generate Random Binary component (0 or 1, scaled)
    // We'll flip a bit every 5 samples to simulate data pulses
    static int bit_val = 0;
    static int pulse_count = 0;
    if (pulse_count++ >= 5) {
      bit_val = random(0, 2);
      pulse_count = 0;
    }
    float data_val = (float)bit_val * 0.5; // Scale data influence
    
    // Combined clean signal
    float clean_signal = sine_val + data_val;
    
    // Send to Serial
    Serial.println(clean_signal, 4);
    
    // Increment time
    t += 1.0 / sampling_rate;
    if (t > 10.0) t = 0; // Reset time to avoid overflow issues over long periods
  }
}
